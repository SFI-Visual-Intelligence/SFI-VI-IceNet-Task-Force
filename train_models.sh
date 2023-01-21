# for i in {1..42}
# do
#     python icenet/train_icenet.py --seed=$i
# done

for i in {42..84}
do
    python icenet/train_icenet.py --seed=$i --dropout_mc
done
