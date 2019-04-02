#! /bin/bash
python3 model_test_accuracy.py --model_path model/premade/semanticsegmentation_frozen_person_quantized_08.tflite --tflite
python3 model_test_accuracy.py --model_path model/1_model/1_model_20e_128_aug_quantized.tflite --tflite
python3 model_test_accuracy.py --model_path model/2_model/2_model_26e_128_quantized.tflite --tflite 
python3 model_test_accuracy.py --model_path model/3_model/3_model_26e_128_quantized.tflite --tflite 
python3 model_test_accuracy.py --model_path model/4_model/4_model_26e_96_128_quantized.tflite --tflite --init_size 96 128 --no_squared
python3 model_test_accuracy.py --model_path model/5_model/5_model_32e_96_128_aug_quantized.tflite --tflite --init_size 96 128 --no_squared
python3 model_test_accuracy.py --model_path model/6_model/6_model_32e_96_128_aug_quantized.tflite --tflite --init_size 96 128 --no_squared

#python3 model_test_accuracy.py --model_path model/1_model/1_model_20e_128.pb --no_tflite
#python3 model_test_accuracy.py --model_path model/2_model/2_model_26e_128.pb --no_tflite
#python3 model_test_accuracy.py --model_path model/3_model/3_model_26e_128.pb --no_tflite
#python3 model_test_accuracy.py --model_path model/4_model/4_model_26e_96_128.pb --no_tflite --init_size 96 128 --no_squared
