use dfdx::{
    prelude::{Adam, LinearConfig, Tanh},
    shapes::Const,
    tensor::{Error, Tensor},
};
use dfdx_core::{
    nn_traits::{BuildModuleExt, Module, Optimizer, ZeroGrads, SaveSafeTensors, LoadSafeTensors},
    prelude::mse_loss,
    shapes::Rank1,
    tensor::{AsArray, Trace},
    tensor_ops::{AdamConfig, Backward},
};
use dfdx_derives::Sequential;

#[cfg(not(feature = "cuda"))]
pub type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
pub type Device = dfdx::tensor::Cuda;


/// Autoencoder consisting of an encoder and decoder network
///
/// R = real-space vector size, L = latent-space vector size
pub struct AutoEncoder<const R: usize, const L: usize> {
    encoder: MLPConfig<R, L>,
    decoder: MLPConfig<L, R>,
    model: Option<(MLP<R, L, f32, Device>, MLP<L, R, f32, Device>)>,
}

/// Multi-layer perceptron with 1 hidden layer
///
/// I = input vector size, O = output vector size
#[derive(Debug, Clone, Sequential)]
#[built(MLP)]
pub struct MLPConfig<const I: usize, const O: usize> {
    linear1: LinearConfig<Const<I>, usize>,
    act1: Tanh,
    linear2: LinearConfig<usize, usize>,
    act2: Tanh,
    linear3: LinearConfig<usize, Const<O>>,
}

impl<const I: usize, const O: usize> MLPConfig<I, O> {
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

impl<const I: usize, const O: usize> MLP<I, O, f32, Device> {
    /// Save the MLP parameters to a folder of `.safetensor` files 
    pub fn save_safetensors(&self, path: &str, id: &str) {
        self.linear1.save_safetensors(path.to_owned()+id+"_linear1.safetensors").expect("Linear1 failed to save");
        self.linear2.save_safetensors(path.to_owned()+id+"_linear2.safetensors").expect("Linear2 failed to save");
        self.linear3.save_safetensors(path.to_owned()+id+"_linear3.safetensors").expect("Linear3 failed to save");
    }

    /// Load the MLP parameters from a folder of `.safetensor` files 
    pub fn load_safetensors(&mut self, path: &str, id: &str) {
        self.linear1.load_safetensors(path.to_owned()+id+"_linear1.safetensors").expect("Linear1 failed to load");
        self.linear2.load_safetensors(path.to_owned()+id+"_linear2.safetensors").expect("Linear2 failed to load");
        self.linear3.load_safetensors(path.to_owned()+id+"_linear3.safetensors").expect("Linear3 failed to load");
    }
}

impl<const R: usize, const L: usize> AutoEncoder<R, L> {
    pub fn new(encoder: MLPConfig<R, L>, decoder: MLPConfig<L, R>) -> Self {
        Self {
            encoder,
            decoder,
            model: None,
        }
    }

    /// Fit the model to the input data
    pub fn fit(&mut self, x: Tensor<(usize, Const<R>), f32, Device>, epochs: usize, lr: f64) {
        // Initialise device
        let dev: Device = Device::default();

        // Define architecture
        let mut model = if let Some(m) = &self.model {
            m.clone()
        } else {
            let arch = (self.encoder.clone(), self.decoder.clone());
            dev.build_module::<f32>(arch)
        };
        //let arch = (self.encoder.clone(), self.decoder.clone()); //let arch = self.encoder.clone();
        //let mut model = dev.build_module::<f32>(arch);
        let mut grads = model.alloc_grads();

        // Initialise optimiser
        let mut opt = Adam::new(
            &model,
            AdamConfig {
                lr,
                ..Default::default()
            },
        );

        //let y: Tensor<(usize, Const<L>), f32, _> = dev.sample_normal_like(&(10_000, Const));

        // Optimisation loop
        for i in 0..epochs {
            // Collect the gradients of the network
            let prediction = model.forward_mut(x.trace(grads));
            let loss = mse_loss(prediction, x.clone()); //let loss = mse_loss(prediction, y.clone());
            println!("Loss after update {i}: {:?}", loss.array());
            grads = loss.backward();

            // Update weights
            opt.update(&mut model, &grads)
                .expect("Oops, there were some unused params");

            model.zero_grads(&mut grads);
        }

        // save model parameters (broken)
        //model.save_safetensors("checkpoint.npz").expect("oopsie");

        self.model = Some(model)
    }

    /// Encode a real-space input vector to latent space
    pub fn encode(
        &self,
        x: Tensor<Rank1<R>, f32, Device>,
    ) -> Option<Tensor<Rank1<L>, f32, Device>> {
        match &self.model {
            Some((enc, _)) => Some(enc.forward(x)),
            None => None,
        }
    }

    /// Decode a latent space vector to real space
    pub fn decode(
        &self,
        x: Tensor<Rank1<L>, f32, Device>,
    ) -> Option<Tensor<Rank1<R>, f32, Device>> {
        match &self.model {
            Some((_, dec)) => Some(dec.forward(x)),
            None => None,
        }
    }

    /// Save the model parameters to a folder of `.safetensor` files 
    pub fn save(&self, path: &str) {
        self.model.as_ref().expect("Encoder model not found").0.save_safetensors(path,"encoder");
        self.model.as_ref().expect("Decoder model not found").1.save_safetensors(path,"decoder");

        //model.0.save_safetensors("params/","encoder");
        //model.1.save_safetensors("params/","decoder");
        //self.model.unwrap().0.save_safetensors("checkpoint.npz")?;
    }

    /// Load the model parameters from a folder of `.safetensor` files 
    pub fn load(&mut self, path: &str) {
        match self.model.as_mut() {
            Some(m) => {
                m.0.load_safetensors(path, "encoder");
                m.1.load_safetensors(path, "decoder");
            },
            None => {
                let dev: Device = Device::default();
                let arch = (self.encoder.clone(), self.decoder.clone());
                let mut m = dev.build_module::<f32>(arch);

                m.0.load_safetensors(path, "encoder");
                m.1.load_safetensors(path, "decoder");

                self.model = Some(m);
            },
        }
    }
}