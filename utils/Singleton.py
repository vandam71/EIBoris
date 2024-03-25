class Singleton(type):
    """A metaclass that creates a Singleton base class when called."""

    _instances = {}  # Dictionary to store instances of Singleton classes

    def __call__(cls, *args, **kwargs):
        """Overrides the __call__ method to enforce singleton behavior.
        Args:
            cls: The class being instantiated.
            *args: Positional arguments passed to the class constructor.
            **kwargs: Keyword arguments passed to the class constructor.
        Returns:
            The instance of the Singleton class.
        """
        if cls not in cls._instances:
            # If the class doesn't have an instance yet, create a new one and store it
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]  # Return the stored instance for subsequent calls
