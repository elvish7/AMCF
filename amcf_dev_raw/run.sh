# gdown https://drive.google.com/uc?id=1V8YTFitg8KKT9_3IOjA1Bx8YWxLRNQbC
# main.py
# delete train data
# gdown
# evaluate
# delete evaluate data
for d in 2019-06-30 #2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
do
    python main.py --date ${d}  | tee ${DIR}/amcf_results.txt
done
# permision denied