
MODELS=("CNNLSTM")
FS=(5)

HS=(512)

for sentiment in true; do
    for F in "${FS[@]}"; do
        for hid in "${HS[@]}"; do

            for MODEL in "${MODELS[@]}"; do
                echo "Running $MODEL with $F features and sentiment=$sentiment"

                python main_Aug.py --MODEL "$MODEL" --n_features "$F" --sentiment "$sentiment" --hidden_dim "$hid"
            done
        done
    done
done