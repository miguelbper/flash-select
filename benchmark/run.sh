uv run benchmark/benchmark.py --m_val 10000  --n 10  --n_estimators 100 --plot_results | tee logs/benchmark_10000_10.log
uv run benchmark/benchmark.py --m_val 25000  --n 25  --n_estimators 100 --plot_results | tee logs/benchmark_25000_25.log
uv run benchmark/benchmark.py --m_val 50000  --n 50  --n_estimators 100 --plot_results | tee logs/benchmark_50000_50.log
uv run benchmark/benchmark.py --m_val 75000  --n 75  --n_estimators 100 --plot_results | tee logs/benchmark_75000_75.log
uv run benchmark/benchmark.py --m_val 100000 --n 100 --n_estimators 100 --plot_results | tee logs/benchmark_100000_100.log
uv run benchmark/benchmark.py --m_val 200000 --n 200 --n_estimators 100 --plot_results | tee logs/benchmark_200000_200.log
