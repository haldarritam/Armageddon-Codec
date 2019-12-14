import concurrent.futures
import time

def test_func(a, b, c, i):
  ret1 = a * i * 1
  ret2 = b * i * 2
  ret3 = c * i * 3

  time.sleep(i % 3)

  return ret1, ret2, ret3

with concurrent.futures.ProcessPoolExecutor() as executor:
  # f1 = executor.submit(test_func, 1, 1, 1, 2)
  # print(f1.result())

  results = [executor.submit(test_func, 1, 1, 1, i) for i in range(5)]

  for f in concurrent.futures.as_completed(results):
    print(f.result())
