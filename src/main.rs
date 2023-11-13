use autoencoder::{Device, MLP, AutoEncoder};

use dfdx::{
    shapes::{Const, Rank1},
    tensor::{SampleTensor, Tensor},
};


fn main() {
    // Initialise device
    let dev: Device = Device::default();

    // Define architecture
    let mut ae: AutoEncoder<784, 10> = AutoEncoder::new(
        MLP::new(400, 48),
        MLP::new(48, 400)
    );

    // Initialise dummy data
    let x = dev.sample_normal_like(&(100, Const::<784>));

    // Fit model
    ae.fit(x);

    let x: Tensor<Rank1<784>, f32, _> = dev.sample_normal();
    let y = ae.encode(x).unwrap();
    println!("{:?}", y.as_vec())

}
/* 
fn main() {
    // Initialise device
    let dev: Device = Device::default();

    // Define architecture
    let ae: AESequence<784, 10> = AESequence {
        encoder: MLP::new(400, 48),
        decoder: MLP::new(48, 400),
    };

    // Initialise dummy data
    let x = dev.sample_normal_like(&(100, Const::<784>));

    // Fit model
    let model = ae.fit(x); 


    let x: Tensor<Rank1<784>, f32, _> = dev.sample_normal();
    let y = model.forward(x);
    println!("{:?}", y.as_vec())
}
*/

