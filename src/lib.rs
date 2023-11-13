use dfdx::{shapes::Const,
    prelude::{LinearConfig, Tanh, Adam},
    tensor::{Error, Tensor},
};
use dfdx_core::{
    prelude::mse_loss,
    nn_traits::{BuildModuleExt, Module, Optimizer, ZeroGrads},
    tensor::{AsArray, Trace},
    tensor_ops::{Backward, AdamConfig}, shapes::Rank1,
};
use dfdx_derives::Sequential;


//pub type Device = dfdx::tensor::Cpu;
pub type Device = dfdx::tensor::Cuda;

/// Autoencoder consisting of an encoder and decoder network
pub struct AutoEncoder<const R: usize, const L: usize> {
    encoder: MLP<R, L>,
    decoder: MLP<L, R>,
    model: Option<(DeviceMLP<R, L, f32, Device>, DeviceMLP<L, R, f32, Device>)>
}


/* 
#[derive(Debug, Clone, Sequential)]
pub struct AEConfig<const R: usize, const L: usize> {
    pub encoder: MLP<R, L>,
    pub decoder: MLP<L, R>,
}
*/

/// Multi-layer perceptron with 1 hidden layer
#[derive(Debug, Clone, Sequential)]
pub struct MLP<const I: usize, const O: usize> {
    linear1: LinearConfig<Const<I>, usize>,
    act1: Tanh,
    linear2: LinearConfig<usize, usize>,
    act2: Tanh,
    linear3: LinearConfig<usize, Const<O>>,
}

impl<const I: usize, const O: usize> MLP<I, O> {
    pub fn new(l1: usize, l2: usize) -> Self {
        Self {
            linear1: LinearConfig::new(Const, l1),
            act1: Default::default(),
            linear2: LinearConfig::new(l1, l2),
            act2: Default::default(),
            linear3: LinearConfig::new(l2, Const),
        }
    }
}

/*
impl<const R: usize, const L: usize> AESequence<R, L> {
    /// Return a model fitted to the input data
    pub fn fit(&self, x: Tensor<(usize, Const<R>), f32, Device>) -> DeviceAESequence<R, L, f32, Device> {
        // Initialise device
        let dev: Device = Device::default();

        // Define architecture
        let arch: AESequence<R, L> = self.clone();
        let mut model = dev.build_module::<f32>(arch);
        let mut grads = model.alloc_grads();

        // Initialise optimiser
        let mut opt = Adam::new(
            &model,
            AdamConfig {
                lr: 1e-3,
                ..Default::default()
            }
        );

        // Optimisation loop
        for i in 0..50 {
            // Collect the gradients of the network
            let prediction = model.forward_mut(x.trace(grads));
            let loss = mse_loss(prediction, x.clone());
            println!("Loss after update {i}: {:?}", loss.array());
            grads = loss.backward();

            //  Update weights
            opt.update(&mut model, &grads)
                .expect("Oops, there were some unused params");
            


            model.zero_grads(&mut grads);
        }

        model
    }
}
*/


impl<const R: usize, const L: usize> AutoEncoder<R, L> {
    pub fn new(encoder: MLP<R, L>, decoder: MLP<L, R>) -> Self {
        Self { encoder, decoder, model: None}
    }

    /// Fit the model to the input data
    pub fn fit(&mut self, x: Tensor<(usize, Const<R>), f32, Device>) {
        // Initialise device
        let dev: Device = Device::default();

        // Define architecture
        let arch = (self.encoder.clone(), self.decoder.clone());//self.ae_seq.clone();
        let mut model = dev.build_module::<f32>(arch);
        let mut grads = model.alloc_grads();

        // Initialise optimiser
        let mut opt = Adam::new(
            &model,
            AdamConfig {
                lr: 1e-3,
                ..Default::default()
            }
        );

        // Optimisation loop
        for i in 0..50 {
            // Collect the gradients of the network
            let prediction = model.forward_mut(x.trace(grads));
            let loss = mse_loss(prediction, x.clone());
            println!("Loss after update {i}: {:?}", loss.array());
            grads = loss.backward();

            //  Update weights
            opt.update(&mut model, &grads)
                .expect("Oops, there were some unused params");
            
            model.zero_grads(&mut grads);
        }

        self.model = Some(model)
    }

    pub fn encode(&self, x: Tensor<Rank1<R>, f32, Device>) -> Option<Tensor<Rank1<L>, f32, Device>> {
        match &self.model {
            Some((enc, _)) => Some(enc.forward(x)),
            None => None,
        }
    }

    pub fn decode(&self, x: Tensor<Rank1<L>, f32, Device>) -> Option<Tensor<Rank1<R>, f32, Device>> {
        match &self.model {
            Some((_, dec)) => Some(dec.forward(x)),
            None => None,
        }
    }
}