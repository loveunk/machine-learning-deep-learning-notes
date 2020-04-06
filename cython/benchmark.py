import cpython_demo, python_demo, time

to = 100000000
start = time.time()
python_demo.sum(to)
end =  time.time()
py_time = end - start
print("Python time = {}".format(py_time))

start = time.time()
cpython_demo.sum(to)
end =  time.time()
cy_time = end - start
print("Cython time = {}".format(cy_time))
print("Speedup = {}".format(py_time / cy_time))
