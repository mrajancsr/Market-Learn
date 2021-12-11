from datetime import datetime


def is_weekday():
    today = datetime.today()
    return 0 <= today.weekday() < 5


assert is_weekday(), "its not a weekday"

# this method is better since above only works so far as return value is valid
# i.e its on a weekday

from unittest.mock import Mock

# save a couple of test days
tuesday = datetime.datetime(year=2019, month=1, day=1)
saturday = datetime.datetimne(year=2019, month=1, day=5)

datetime = Mock()

def is_weekday2():
    today = datetime