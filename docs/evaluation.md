# Fast evalution

To reproduce the paper results, either by retraining a network or by using a pre-trained model, please follow these steps.

### Data preparation

* Please download the data and pre-trained models by run python file:
  * ```bash
    python evaluation/get_data.py
    ```
* Then please uncompress all`tar.gz` file by run this command.
  * ```bash
    # Make the above loop a function to be called more than once 
    # In case of compressed files inside compressed files this will 
    # run twice the previous snippet.

    $ function decompress_all_complete () {
         function decompress_all () {
              for file in `find *`; do
                    sudo tar -xvf "${file}" ; done
         } ;
        for i in `echo {1..2}`; do
             decompress_all_complete; done
    }
    ```

### Evaluation

* Execute instructions in evaluation/evaluation.sh
  * ```bash
    python train.py \
      -heterodir "${data_dir}/aliens" \
      -load_Path "${save_path_of_pretrained_model}$/best_model_alien_hgt" \
      -writerdir "${save_path_of_middle_data}/aliens$" \
      -losstype "focal" \
      -n_hid 120  \
      -n_layers 4
    ```
