DIR=results
## Evaluation span: 1 month
len=1m
for w in 1 #6 1 #10 16 32 64 128 256
do
for f in 0.9 #0.0000001
do
## Pure CF no side information
for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
do
    python3 local_lightfm_main.py --date ${d} --train w_103_${d}.csv --evaluation evaluation_df_${d}.csv --decay_factor ${f} --decay_window ${w} | grep 'Today' | awk -F' ' -vf=${f} -vw=${w} '{print w,f,$2,$5}' >> ${DIR}/lightfm_pure_cf_results.txt
done
done
echo " " >> ${DIR}/lightfm_pure_cf_results.txt
done

### With User Features
#for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
#do
#    python3 lightfm_main.py --date ${d} --eval_duration ${len} --user_ft | grep 'Today' | awk -F' ' '{print $2,$5}' >> ${DIR}/lightfm_cf_with_user_results.txt
#done
#echo " " >> ${DIR}/lightfm_cf_with_user_results.txt
### With Item Features
#for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
#do
#    python3 lightfm_main.py --date ${d} --eval_duration ${len} --item_ft | grep 'Today' | awk -F' ' '{print $2,$5}' >> ${DIR}/lightfm_cf_with_item_results.txt
#done
#echo " " >> ${DIR}/lightfm_cf_with_item_results.txt
### With User & Item Features
#for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
#do
#    python3 lightfm_main.py --date ${d} --eval_duration ${len} --user_ft --item_ft | grep 'Today' | awk -F' ' '{print $2,$5}' >> ${DIR}/lightfm_cf_with_user_item_results.txt
#done
#echo " " >> ${DIR}/lightfm_cf_with_user_item_results.txt
