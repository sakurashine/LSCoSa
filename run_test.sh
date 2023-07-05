array1=("1")  
array2=( "1_Dy4hot+1.6KL"  )
array3=( 0.4  )
array4=( 2  )
length=${#array1[@]}
for ((i=0;i<$length;i++))
do
python linear.py   \
    -a resnet18 \
    --lr 2 \
    --batch_size 64 \
    --pretrained /root/DJ/LSCoSa/train_log/IndianPines/Type_0/lr_0.3_0.02/memlr_0.3_0.02/t_0.08_memt0.08/wd_1.5e-06_memwd0.0/mocomdecay_0.99/memgradm_0.9/hidden128_out128/batch_32/epoch_100/warm_5/time_1/model_best.pth.tar \
    --epochs 200 \
    --folder "../dataset/IndianPines" \
    --dataset "IndianPines" \
    --patch_size 15 \
    --class_balancing  \
    --run 1 \
    --load_data 0.10  \
    --fine_tune no  \
    --desc ${array2[$i]} \
    --rho ${array3[$i]} \
    --norm ${array4[$i]}
done