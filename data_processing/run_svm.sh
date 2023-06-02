# IndianPines
# array1=("40" "41" "42" "43" "44" "45" "46" "47" "48")
# array2=("AAC_caco" "AAD_4hot" "AAE_caco+memax" "AAF_4hot+memax" "AAG_Dy4hot" "AAH_caco+repetition" "AAI_Dy4hot+repetition" "AAJ_4hot+repetition" "AAK_Dy4hot+memax")
# length=${#array1[@]}
# for ((i=0;i<$length;i++))
# do
# python svm.py   \
#     -a resnet18 \
#     --lr 2 \
#     --batch_size 64 \
#     --pretrained /home/lmm/DJ/caco/train_log/IndianPines/Type_0/lr_0.01_0.01/memlr_3.0_3.0/t_0.08_memt0.08/wd_1.5e-06_memwd0.0/mocomdecay_0.99/memgradm_0.9/hidden128_out128/batch_32/epoch_200/warm_5/time_${array1[$i]}/model_best.pth.tar \
#     --dist-url 'tcp://localhost:10002' \
#     --epochs 1 \
#     --world-size 1 \
#     --rank 0 \
#     --folder "../dataset/IndianPines" \
#     --dataset "IndianPines" \
#     --patch_size 9 \
#     --class_balancing  \
#     --run 1 \
#     --load_data 0.01 \
#     --fine_tune no  \
#     --desc ${array2[$i]}
# done

# PaviaU
array1=("1" "2" "3")
array2=("1_caco" "2_caco+memax" "3_Dy4hot+memax")
array3=("20" "100" "0.03")
length=${#array1[@]}
for ((i=0;i<$length;i++))
do
python svm.py   \
    -a resnet18 \
    --lr 2 \
    --batch_size 64 \
    --pretrained /home/lmm/DJ/caco/train_log/PaviaU/Type_0/lr_0.02_0.02/memlr_3.0_3.0/t_0.08_memt0.08/wd_1.5e-06_memwd0.0/mocomdecay_0.99/memgradm_0.9/hidden128_out128/batch_32/epoch_200/warm_5/time_${array1[$i]}/model_best.pth.tar \
    --dist-url 'tcp://localhost:10002' \
    --epochs 1 \
    --world-size 1 \
    --rank 0 \
    --folder "../dataset/PaviaU" \
    --dataset "PaviaU" \
    --patch_size 15 \
    --class_balancing  \
    --run 1 \
    --load_data 20 \
    --fine_tune no  \
    --desc ${array2[$i]}
done