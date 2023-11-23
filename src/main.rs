use autoencoder::{AutoEncoder, Device, MLPConfig};

use std::{error::Error, fs::File};
use dfdx::{
    shapes::Const,
    tensor::Tensor,
};
use dfdx_core::tensor::TensorFromVec;
use rand::prelude::SliceRandom;


fn main() -> Result<(), Box<dyn Error>> {
    // Initialise device & architecture
    let dev: Device = Device::default();
    let mut ae: AutoEncoder<784, 10> =
        AutoEncoder::new(MLPConfig::new(400, 150), MLPConfig::new(150, 400));

    // Load dataset
    let (x_train, x_val) = read_mnist(0.20)?;
    let (x_train_size, x_val_size) = (x_train.len() / 784, x_val.len() / 784);

    println!("train size: {x_train_size}, validation size: {x_val_size}");

    let x_train: Tensor<(usize, Const<784>), f32, _> = dev.tensor_from_vec(x_train, (x_train_size, Const));
    let x_val: Tensor<(usize, Const<784>), f32, _> = dev.tensor_from_vec(x_val, (x_val_size, Const));

    // Fit model
    ae.partial_fit(x_train, 100, 5.0e-4);

    println!("Validation loss: {}", ae.calc_validation_loss(x_val));
    
    // ae.save(&Path::new("params/"), "model")?;
    
    // Load model
    /*
    ae.load(
        &Path::new("params/model_encoder.safetensors"),
        &Path::new("params/model_decoder.safetensors"),
    )?;
    */

    // Print results
    /* Validation input
    let x: Tensor<Rank1<784>, f32, _> =
        dev.tensor_from_vec((0..784).into_iter().map(|x| x as f32).collect(), (Const,));
    let y = ae.encode(x).unwrap();
    println!("{:?}", y.as_vec())
    */

    /* Principal components
    let x: Tensor<Rank1<10>, f32, _> = dev.tensor_from_vec(vec![0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], (Const,));
    let y = ae.decode(x).unwrap();
    y.as_vec().iter().for_each(|i| println!("{:?},", i));
    */

    /*
    let x1 = read_mnist().ok().unwrap()[3*784..4*784].to_vec();

    let x: Tensor<Rank1<784>, f32, _> = dev.tensor((x1, (Const,)));
    let y = ae.decode(ae.encode(x).unwrap()).unwrap();

    y.as_vec().iter().for_each(|i| println!("{:?},", i));
    */

    Ok(())
}

fn read_mnist(val_split: f64) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let file = File::open("mnist_test.csv")?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    // Generate pixel arrays
    let mut pixel_arrays: Vec<Vec<f32>> = rdr
        .deserialize()
        .filter_map(|record: Result<Vec<f32>, _>| match record {
            Ok(record) => {
                let mut vec: Vec<f32> = record.iter().map(|x| x / 255.).collect();
                vec.remove(0);
                Some(vec)
            }
            Err(e) => {
                eprintln!("Error parsing record: {}", e);
                None
            }
        })
        .collect();

    // Shuffle vector and split
    let mut rng = rand::thread_rng();
    pixel_arrays.shuffle(&mut rng);

    //let training_set = &pixel_arrays[0..(val_split as usize * pixel_arrays.len())];
    //let training_set: Vec<f32> = (0..(val_split as usize * pixel_arrays.len())).into_iter().map(|i| pixel_arrays[i]).flatten().collect();

    let validation_set: Vec<f32> = pixel_arrays.drain(0..((pixel_arrays.len() as f64 * val_split) as usize)).flatten().collect();
    let training_set: Vec<f32> = pixel_arrays.into_iter().flatten().collect();

    Ok((training_set, validation_set))
}
