array1=( "1"  )
array2=( "Dy4hot+1.6KL" )
array3=( 0.4 )
length=${#array1[@]}
for ((i=0;i<$length;i++))
do  
python main.py \
    --lr=0.3 \
    --lr_final=0.02 \
    --memory_lr=0.3 \
    --memory_lr_final=0.02 \
    --cluster=256 \
    --moco_t=0.08 \
    --mem_t=0.08 \
    --data_folder=../dataset/IndianPines \
    --dataset=IndianPines \
    --batch_size=32 \
    --epochs=2 \
    --patch_size=15 \
    --time ${array1[$i]}  \
    --loss ${array2[$i]}  \
    --rho ${array3[$i]}  \
    --norm=2  \
    --seed 16
done
