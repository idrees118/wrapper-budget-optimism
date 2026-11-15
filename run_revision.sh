for v in k1 k11 a90 a95 pop20 pop50; do
  python src/run_experiments.py --config configs/sens_$v.yaml --methods ga pso random --out_tag sens_$v 2>&1 | tee logs/sens_$v.log
done
python src/run_experiments.py --methods gwo woa --out_tag newopt 2>&1 | tee logs/newopt.log
