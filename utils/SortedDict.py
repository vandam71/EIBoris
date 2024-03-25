from typing import Any


class SortedDict:
    """A dictionary-like object that maintains sorted keys based on their associated values."""

    def __init__(self) -> None:
        """Initialize an empty SortedDict."""
        self._dict = {}
        self._keys = []

    def __len__(self) -> int:
        """Return the number of key-value pairs in the SortedDict."""
        return len(self._dict)

    def __getitem__(self, key: str) -> Any:
        """Return the value associated with the given key.
        Args:
            key (str): The key to retrieve the value for.
        Returns:
            Any: The value associated with the key.
        Raises:
            KeyError: If the key is not found in the SortedDict.
        """
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value associated with the given key.
        If the key already exists, its associated value is updated.
        If the key is new, it is added to the SortedDict.
        Args:
            key (str): The key to set the value for.
            value (Any): The value to associate with the key.
        """
        if key not in self._dict:
            self._keys.append(key)
        self._dict[key] = value
        self._keys.sort(key=lambda x: self._dict[x], reverse=True)

    def __delitem__(self, key: str) -> None:
        """Remove the key-value pair with the given key from the SortedDict.
        Args:
            key (str): The key to remove.
        Raises:
            KeyError: If the key is not found in the SortedDict.
        """
        del self._dict[key]
        self._keys.remove(key)

    def keys(self) -> list:
        """Return a list of keys in the SortedDict, in the sorted order.
        Returns:
            list: A list of keys.
        """
        return self._keys.copy()

    def values(self) -> list:
        """Return a list of values in the SortedDict, in the order of their corresponding keys.
        Returns:
            list: A list of values.
        """
        return [self._dict[key] for key in self._keys]

    def items(self) -> list[tuple]:
        """Return a list of key-value pairs in the SortedDict, in the sorted order.
        Returns:
            list[tuple]: A list of key-value pairs.
        """
        return [(key, self._dict[key]) for key in self._keys]

    def __repr__(self) -> str:
        """Return a string representation of the SortedDict.
        Returns:
            str: A string representation of the SortedDict.
        """
        items = [(key, value) for key, value in self.items()]
        return str(items)
