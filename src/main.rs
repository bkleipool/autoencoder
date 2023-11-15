use autoencoder::{AutoEncoder, Device, MLPConfig};

use std::{error::Error, fs::File, path::Path};
use dfdx::{
    shapes::{Const, Rank1},
    tensor::Tensor,
};
use dfdx_core::tensor::{TensorFrom, TensorFromVec};

fn main() {
    // Initialise device
    let dev: Device = Device::default();

    // Define architecture
    let mut ae: AutoEncoder<784, 10> =
        AutoEncoder::new(MLPConfig::new(400, 100), MLPConfig::new(100, 400));

    // Initialise data
    let x: Tensor<(usize, Const<784>), f32, _> =
        dev.tensor((read_mnist().ok().unwrap(), (10000, Const)));

    // Fit model
    ae.fit(x, 50, 1.0e-3);
    let _ = ae.save(&Path::new("params/"), "model");
    /* 
    let _ = ae.load(
        &Path::new("params/model_encoder.safetensors"),
        &Path::new("params/model_decoder.safetensors"),
    );
    */

    // Print results
    let x: Tensor<Rank1<784>, f32, _> =
        dev.tensor_from_vec((0..784).into_iter().map(|x| x as f32).collect(), (Const,)); //dev.sample_normal();
    let y = ae.encode(x).unwrap();
    println!("{:?}", y.as_vec())

    /*
    let x: Tensor<Rank1<10>, f32, _> = dev.tensor_from_vec(vec![0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], (Const,));
    let y = ae.decode(x).unwrap();
    y.as_vec().iter().for_each(|i| println!("{:?},", i));
    */
}

fn read_mnist() -> Result<Vec<f32>, Box<dyn Error>> {
    let file = File::open("mnist_test.csv")?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let pixel_values: Vec<f32> = rdr
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
        .flatten()
        .collect();

    Ok(pixel_values)
}
