import os
import time
import logging

logs_folder = "logs/"
benchmark_folder = "paper_benchmarks/"
mathsat_path = "path/to/mathsat"
cvc5_path = "path/to/cvc5"
yices_path = "path/to/yices"
z3_path = "path/to/z3"
test_only_relu = True

logger_out = logging.getLogger("exp_launcher_out")
logger_exp = logging.getLogger("exp_launcher_exp")

if not os.path.exists(logs_folder):
    os.mkdir(logs_folder)

file_handler_out = logging.FileHandler("logs/exp_out.txt")
file_handler_exp = logging.FileHandler("logs/exp_res.txt")

file_handler_out.setLevel(logging.INFO)
file_handler_exp.setLevel(logging.INFO)

logger_out.addHandler(file_handler_out)
logger_exp.addHandler(file_handler_exp)

logger_exp.setLevel(logging.INFO)
logger_out.setLevel(logging.INFO)

benchmarks = sorted(os.listdir(benchmark_folder))
mathsat_benchmarks = [x for x in benchmarks if "mathsat" in x]
cvc5_benchmarks = [x for x in benchmarks if "cvc" in x]

if test_only_relu:
    mathsat_benchmarks = mathsat_benchmarks[0:15]
    cvc5_benchmarks = cvc5_benchmarks[0:15]

timeout = 3600
sys_timeout = timeout + 5

##### MATHSAT QUERIES #####
for b in mathsat_benchmarks:

    logger_out.info(f"\nMATHSAT {b}")
    cmd = f"timeout {timeout}s {mathsat_path} -verbosity=2 -stats -model {benchmark_folder}{b} >> logs/exp_out.txt"
    start = time.perf_counter()
    res = os.system(cmd)
    stop = time.perf_counter()
    logger_exp.info(f"mathsat,{b},{stop - start}")

##### CVC5 QUERIES #####
for b in cvc5_benchmarks:

    logger_out.info(f"\nCVC5 {b}")
    cmd = f"timeout {timeout}s {cvc5_path} --verbose --stats {benchmark_folder}{b} >> logs/exp_out.txt"
    start = time.perf_counter()
    res = os.system(cmd)
    stop = time.perf_counter()
    logger_exp.info(f"cvc5,{b},{stop - start}")

##### Z3 QUERIES #####
for b in mathsat_benchmarks:

    logger_out.info(f"\nZ3 {b}")
    cmd = f"timeout {sys_timeout}s {z3_path} -smt2 -model -v:5 -T:{timeout} -st {benchmark_folder}{b} >> logs/exp_out.txt"
    start = time.perf_counter()
    res = os.system(cmd)
    stop = time.perf_counter()
    logger_exp.info(f"z3,{b},{stop - start}")

##### YICES QUERIES #####
for b in mathsat_benchmarks:

    logger_out.info(f"\nYICES {b}")
    cmd = f"timeout {sys_timeout}s {yices_path} -t {timeout} -s -v 10 --smt2-model-format {benchmark_folder}{b} >> logs/exp_out.txt"
    start = time.perf_counter()
    res = os.system(cmd)
    stop = time.perf_counter()
    logger_exp.info(f"yices,{b},{stop - start}")
