from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    # implement a tree data structure that stores named :class:minitorch.Parameter on each node.
    # Such a data structure makes it easy for users to create trees that can be walked to find all
    # of the parameters of interest.
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.


    Attributes
    ----------
        _modules: Storage of the child modules
        _parameters: Storage of the module's parameters
        training: Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the `training` flag of this and descendent to true."""
        self.training = True
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        """Set the `training` flag of this and descendent to false."""
        self.training = False
        for module in self.modules():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Recursively collects all parameters and their names in this module and its descendants.

        The names of parameters from child modules are prefixed with the name of the parent module,
        ensuring unique identification of each parameter in the tree.

        Returns
        -------
            Sequence[Tuple[str, Parameter]]: A sequence of tuples, each containing the name of a parameter
            and the parameter itself.

        """
        parameters = []
        # Collect parameters of the current module
        for name, param in self._parameters.items():
            parameters.append((name, param))

        # Recursively collect parameters from child modules
        for module_name, module in self._modules.items():
            child_params = module.named_parameters()
            for child_name, child_param in child_params:
                parameters.append((f"{module_name}.{child_name}", child_param))
        return parameters

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents.

        Returns
        -------
            Sequence[Parameter]: A sequence of parameters.

        """
        return [param for _, param in self.named_parameters()]

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        """Set an attribute of the module.

        Args:
        ----
            key: The name of the attribute to set.
            val: The value of the attribute to set.

        """
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Retrieve a parameter or module by name.

        Args:
        ----
            key: The name of the parameter or module to retrieve.

        Returns:
        -------
            Any: The parameter or module with the given name, if it exists.

        """
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call self as a function, invoking the forward method.

        Args:
        ----
            *args: Positional arguments passed to forward.
            **kwargs: Keyword arguments passed to forward.

        Returns:
        -------
            Any: Result of the forward method.

        """
        return self.forward(*args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        """Return a string representation of the module.

        Returns
        -------
            str: The string representation of the module.

        """

        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]  # type: ignore
            s = "\n".join(s2)
            s = first + "\n" + s  # type: ignore
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"  # type: ignore

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Initialize the parameter.

        Args:
        ----
            x: The value of the parameter.
            name: The name of the parameter.

        """
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value.

        Args:
        ----
            x: The new value for the parameter.

        """
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
