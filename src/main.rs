use std::{error::Error, fs::File};

use autoencoder::{AutoEncoder, Device, MLPConfig};

use dfdx::{
    shapes::{Const, Rank1},
    tensor::{SampleTensor, Tensor},
};
use dfdx_core::tensor::TensorFrom;

fn main() {
    // Initialise device
    let dev: Device = Device::default();

    // Define architecture
    let mut ae: AutoEncoder<784, 10> = AutoEncoder::new(
        MLPConfig::new(183, 48),
        MLPConfig::new(48, 183)
    );

    // Initialise data
    //let x = dev.sample_normal_like(&(100, Const::<784>));
    let x: Tensor<(usize, Const<784>), f32, _> =
        dev.tensor((read_mnist().ok().unwrap(), (10000, Const)));

    // Fit model
    //ae.fit(x, 50, 6.0e-4);
    ae.load("params/");

    let x: Tensor<Rank1<784>, f32, _> = dev.sample_normal();
    let y = ae.encode(x).unwrap();
    println!("{:?}", y.as_vec())
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
