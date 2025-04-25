


class Quantity:
    def __init__(self, value: float, unit: str) -> None:
        """
        Initialize the Quantity class with a value and a unit.

        :param value: The value of the quantity.
        :param unit: The unit of the quantity.
        """
        self.value = value
        self.unit = unit

    # addition
    def __add__(self, other: "Quantity") -> "Quantity":
        """
        Add two quantities together.

        :param other: The other quantity to add.
        :return: The resulting quantity.
        """
        assert self.unit == other.unit, "Units must be the same to add quantities"
        
        return Quantity(self.value + other.value, self.unit) 
    
    def __sub__(self, other: "Quantity") -> "Quantity":
        """
        Subtract two quantities.

        :param other: The other quantity to subtract.
        :return: The resulting quantity.
        """
        assert self.unit == other.unit, "Units must be the same to subtract quantities"
        
        return Quantity(self.value - other.value, self.unit)
    
    def __neg__(self) -> "Quantity":
        """
        Negate the quantity.

        :return: The negated quantity.
        """
        return Quantity(-self.value, self.unit)
    
    # comparisons
    def __eq__(self, other: "Quantity") -> bool:
        """
        Compare two quantities.

        :param other: The other quantity to compare.
        :return: True if self is equal to other, False otherwise.
        """
        assert self.unit == other.unit, "Units must be the same to compare quantities"
        
        return self.value == other.value
    
    def __ne__(self, other: "Quantity") -> bool:
        """
        Compare two quantities.

        :param other: The other quantity to compare.
        :return: True if self is not equal to other, False otherwise.
        """
        assert self.unit == other.unit, "Units must be the same to compare quantities"
        
        return self.value != other.value

    def __lt__(self, other: "Quantity") -> bool:
        """
        Compare two quantities.

        :param other: The other quantity to compare.
        :return: True if self is less than other, False otherwise.
        """
        assert self.unit == other.unit, "Units must be the same to compare quantities"
        
        return self.value < other.value
    
    def __le__(self, other: "Quantity") -> bool:
        """
        Compare two quantities.

        :param other: The other quantity to compare.
        :return: True if self is less than or equal to other, False otherwise.
        """
        assert self.unit == other.unit, "Units must be the same to compare quantities"
        
        return self.value <= other.value
 
    def __gt__(self, other: "Quantity") -> bool:
        """
        Compare two quantities.

        :param other: The other quantity to compare.
        :return: True if self is greater than other, False otherwise.
        """
        assert self.unit == other.unit, "Units must be the same to compare quantities"
        
        return self.value > other.value
    
    def __ge__(self, other: "Quantity") -> bool:
        """
        Compare two quantities.

        :param other: The other quantity to compare.
        :return: True if self is greater than or equal to other, False otherwise.
        """
        assert self.unit == other.unit, "Units must be the same to compare quantities"
        
        return self.value >= other.value
