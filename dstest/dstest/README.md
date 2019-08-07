python -m dstest.preprocess.import_image  --input_path inputs/mnist --output_path datas/mnist
python -m dstest.preprocess.preprocess  --input_path datas/mnist --output_path outputs/mnist --image_column=image --target_column=x --target_datauri_column=x.data --target_image_size=28x28
python -m dstest.tensorflow.mnist_test