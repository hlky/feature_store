import contextlib
import enum
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureType(enum.Enum):
    INPUT = enum.auto()
    OUTPUT = enum.auto()


@dataclass
class ActionOrder(enum.Enum):
    A_B = enum.auto()
    B_A = enum.auto()


@dataclass
class ActionMathType(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    DIV = enum.auto()
    MUL = enum.auto()


@dataclass
class ActionConcatType(enum.Enum):
    ZEROS = enum.auto()
    ONES = enum.auto()
    COPY = enum.auto()


@dataclass
class ActionConcat:
    order: ActionOrder
    dim: int
    batch: ActionConcatType


@dataclass
class ActionStoredMath:
    type: ActionMathType
    order: ActionOrder


@dataclass
class ActionChunk:
    chunks: int
    dim: int
    index: Optional[int]


@dataclass
class ActionReplace:
    enabled: bool


@dataclass
class ActionStore:
    enabled: bool


@dataclass
class ActionShapes:
    enabled: bool


@dataclass
class ActionConstantMath:
    order: ActionOrder
    type: ActionMathType
    constant: int | float


ActionType = (
    ActionConcat
    | ActionStoredMath
    | ActionReplace
    | ActionStore
    | ActionShapes
    | ActionConstantMath
)

ActionStoredType = ActionConcat | ActionStoredMath | ActionReplace


@dataclass
class FeaturePatternType(enum.Enum):
    FULL = enum.auto()
    START = enum.auto()
    END = enum.auto()
    IN = enum.auto()
    REGEX = enum.auto()


@dataclass
class FeatureConfig:
    type: FeatureType
    pattern_type: Optional[FeaturePatternType]
    actions: List[ActionType]


@dataclass
class FeatureStoreItem:
    pattern: str
    config: FeatureConfig


class FeatureStore:
    skip = ["dropout", "nonlinearity", "act", "conv_act"]

    def __init__(self, features: dict[str, torch.Tensor] = {}):
        self.stack = []
        self.features: dict[str, torch.Tensor] = features
        self.wrapped = set()

    @contextlib.contextmanager
    def module_scope(self, name: str):
        # this fixes some edge cases
        if len(self.stack) > 0 and (
            name.startswith(self.stack[-1] or name.endswith(self.stack[-1]))
        ):
            _ = self.stack.pop()
            self.stack.append(name)
        else:
            self.stack.append(name)
        try:
            yield
        finally:
            if len(self.stack) > 0:
                self.stack.pop()

    def full_scope(self):
        return ".".join(self.stack)

    def add_feature(self, key: str, feature: torch.Tensor):
        self.features.update({key: feature})

    def get_feature(self, key: str):
        return self.features.get(key, None)

    def match_features(
        self, key: str, wishlist: List[FeatureStoreItem], required_type: FeatureType
    ) -> List[FeatureStoreItem]:
        desired_features = []
        for feature in wishlist:
            config = feature.config
            if required_type.name != config.type.name:
                continue
            pattern = feature.pattern
            if (
                (config.pattern_type == FeaturePatternType.FULL and key == pattern)
                or (
                    config.pattern_type == FeaturePatternType.START
                    and key.startswith(pattern)
                )
                or (
                    config.pattern_type == FeaturePatternType.END
                    and key.endswith(pattern)
                )
                or (config.pattern_type == FeaturePatternType.IN and pattern in key)
                or (
                    config.pattern_type == FeaturePatternType.REGEX
                    and re.search(pattern, key) is not None
                )
                or config.pattern_type is None
            ):
                desired_features.append(feature)
        return desired_features

    def store(
        self,
        module: nn.Module,
        prefix: str = "",
        feature_wishlist: List[FeatureStoreItem] = [],
    ):
        named_children = list(module.named_children())
        if named_children:
            for name, child in named_children:
                if name in self.skip:
                    continue
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.ModuleList):
                    for i, sub_child in enumerate(child):
                        self.store(sub_child, f"{full_name}.{i}", feature_wishlist)
                else:
                    self.wrap(child, full_name, feature_wishlist)
                    self.store(child, full_name, feature_wishlist)
        else:
            if not isinstance(module, (nn.Dropout)):
                self.wrap(module, prefix, feature_wishlist)

    def process_action(
        self,
        tensor: torch.Tensor,
        module: nn.Module,
        action: ActionType,
        existing_feature: Optional[torch.Tensor],
        module_key: str,
        feature: FeatureStoreItem,
    ):
        if isinstance(action, ActionStoredType):
            if existing_feature is None:
                logger.info(
                    f"{action.__class__} requested but stored featured not found for key: {module_key}. Feature: {feature}"
                )
                return tensor
        if isinstance(action, ActionConcat):
            if existing_feature is None:
                return tensor
            if action.dim > tensor.ndim:
                logger.info(
                    f"{action.__class__} requested but requested rank {action.dim} > tensor's rank {tensor.ndim} for key: {module_key}. Feature: {feature}"
                )
                return tensor
            batch = tensor.shape[0]
            feature_batch = existing_feature.shape[0]
            if batch < feature_batch:
                number_of_copies = feature_batch - batch
                if action.batch == ActionConcatType.ZEROS:
                    copy = torch.zeros_like(tensor)
                elif action.batch == ActionConcatType.ONES:
                    copy = torch.ones_like(tensor)
                else:
                    copy = tensor
                copies = [copy for _ in range(number_of_copies)]
                if action.order == ActionOrder.A_B:
                    copies.append(tensor)
                else:
                    copies = [tensor].extend(copies)
                tensor = torch.cat(copies)
            elif feature_batch < batch:
                number_of_copies = batch - feature_batch
                if action.batch == ActionConcatType.ZEROS:
                    copy = torch.zeros_like(existing_feature)
                elif action.batch == ActionConcatType.ONES:
                    copy = torch.ones_like(existing_feature)
                else:
                    copy = existing_feature
                copies = [copy for _ in range(number_of_copies)]
                if action.order == ActionOrder.A_B:
                    copies.append(existing_feature)
                else:
                    copies = [existing_feature].extend(copies)
                existing_feature = torch.cat(copies)
            if action.order == ActionOrder.A_B:
                return torch.cat(
                    [tensor, existing_feature],
                    dim=action.dim,
                )
            else:
                return torch.cat(
                    [existing_feature, tensor],
                    dim=action.dim,
                )
        elif isinstance(action, ActionStoredMath):
            if existing_feature is None:
                return tensor
            if action.type == ActionMathType.ADD:
                return tensor + existing_feature
            elif action.type == ActionMathType.SUB:
                if action.order == ActionOrder.A_B:
                    return tensor - existing_feature
                else:
                    return existing_feature - tensor
            elif action.type == ActionMathType.MUL:
                return tensor * existing_feature
            elif action.type == ActionMathType.DIV:
                if action.order == ActionOrder.A_B:
                    return tensor / existing_feature
                else:
                    return existing_feature / tensor
        elif isinstance(action, ActionReplace):
            if existing_feature is None:
                return tensor
            if action.enabled:
                # clone is required if using inference mode tensor in training mode
                if module.training:
                    if existing_feature.requires_grad == False:
                        output = existing_feature.clone()
                        output.requires_grad = True
                        return output
                else:
                    output = existing_feature
                    output.requires_grad = False
                    output.grad = None
                    output.grad_fn = None
                    return output
        elif isinstance(action, ActionChunk):
            chunked = torch.chunk(tensor, action.chunks, action.dim)
            if action.index:
                return chunked[action.index]
            else:
                return chunked
        elif isinstance(action, ActionStore):
            if action.enabled:
                self.add_feature(module_key, tensor)
            return tensor
        elif isinstance(action, ActionShapes):
            if action.enabled:
                logger.info(
                    f"[{feature.config.type.name}] {module_key}: {list(tensor.shape)}"
                )
            return tensor
        elif isinstance(action, ActionConstantMath):
            if action.type == ActionMathType.ADD:
                return tensor + action.constant
            elif action.type == ActionMathType.SUB:
                if action.order == ActionOrder.A_B:
                    return tensor - action.constant
                else:
                    return action.constant - tensor
            elif action.type == ActionMathType.MUL:
                return tensor * action.constant
            elif action.type == ActionMathType.DIV:
                if action.order == ActionOrder.A_B:
                    return tensor / action.constant
                else:
                    return action.constant / tensor
        else:
            logger.debug(
                f"unknown action type: {action.__class__}. Action: {action}. Feature: {feature}. Key: {module_key}."
            )
            return tensor

    def process_features(
        self,
        tensor: torch.Tensor,
        module: nn.Module,
        module_key: str,
        wishlist: List[FeatureStoreItem],
        required_type: FeatureType,
    ):
        features = self.match_features(module_key, wishlist, required_type)
        if len(features) == 0:
            return tensor
        existing_feature = self.get_feature(module_key)
        for feature in features:
            for action in feature.config.actions:
                tensor = self.process_action(
                    tensor=tensor,
                    module=module,
                    action=action,
                    existing_feature=existing_feature,
                    module_key=module_key,
                    feature=feature,
                )
        return tensor

    def wrap(
        self, module: nn.Module, name: str, feature_wishlist: List[FeatureStoreItem]
    ):
        if name in self.wrapped:
            return
        original_forward = module.forward

        module_scope = self.module_scope(name)
        full_scope = self.full_scope
        process_features = self.process_features

        def forward(self, *args, **kwargs):
            with module_scope:
                module_key = full_scope()
                logger.debug(f"Forward pass at: {module_key}")
                arguments = list(args)
                arguments[0] = process_features(
                    tensor=arguments[0],
                    module=module,
                    module_key=module_key,
                    wishlist=feature_wishlist,
                    required_type=FeatureType.INPUT,
                )
                output: torch.Tensor = original_forward(*arguments, **kwargs)
                output = process_features(
                    tensor=output,
                    module=module,
                    module_key=module_key,
                    wishlist=feature_wishlist,
                    required_type=FeatureType.OUTPUT,
                )

                return output

        module.forward = forward.__get__(module, module.__class__)
        self.wrapped.add(name)


class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        return self.conv(x)


class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([A() for _ in range(5)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Main(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = A()
        self.b = B()
        self.layers = nn.ModuleList([B() for _ in range(3)])

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        for layer in self.layers:
            x = layer(x)
        return x


feature_store = FeatureStore()

main = Main().eval()

feature_store.store(
    main,
    feature_wishlist=[
        FeatureStoreItem(
            pattern="",
            config=FeatureConfig(
                type=FeatureType.INPUT,
                pattern_type=None,
                actions=[ActionShapes(enabled=True)],
            ),
        ),
    ],
)
input_tensor = torch.randn(1, 64, 64, 64)
output = main(input_tensor)
