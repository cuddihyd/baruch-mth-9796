from nyt import _iter_month_params

def test_iter_month_params():
  actual = list(_iter_month_params(2020, 6, 30))
  assert actual[0] == (2020, 6)
  assert actual[6] == (2020, 12)
  assert actual[7] == (2021, 1)
