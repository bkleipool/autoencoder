use std::path::Path;

use dfdx::{
    prelude::{Adam, LinearConfig, Tanh},
    shapes::Const,
    tensor::{Error, Tensor},
};
use dfdx_core::{
    nn_traits::{BuildModuleExt, LoadSafeTensors, Module, Optimizer, SaveSafeTensors, ZeroGrads},
    prelude::mse_loss,
    shapes::Rank1,
    tensor::{AsArray, Trace},
    tensor_ops::{AdamConfig, Backward},
};
use dfdx_derives::Sequential;
use safetensors::{tensor::TensorView, SafeTensorError};

//#[cfg(not(feature = "cuda"))]
//pub type Device = dfdx::tensor::Cpu;

//#[cfg(feature = "cuda")]
pub type Device = dfdx::tensor::Cuda;


/// Autoencoder consisting of an encoder and decoder network
///
/// * R = real-space vector size
/// * L = latent-space vector size
pub struct AutoEncoder<const R: usize, const L: usize> {
    encoder: MLPConfig<R, L>,
    decoder: MLPConfig<L, R>,
    model: Option<(MLP<R, L, f32, Device>, MLP<L, R, f32, Device>)>,
}

/// Multi-layer perceptron with 1 hidden layer
///
/// * I = input vector size
/// * O = output vector size
#[derive(Debug, Clone, Sequential)]
#[built(MLP)]
pub struct MLPConfig<const I: usize, const O: usize> {
    linear1: LinearConfig<Const<I>, usize>,
    act1: Tanh,
    linear2: LinearConfig<usize, usize>,
    act2: Tanh,
    linear3: LinearConfig<usize, Const<O>>,
    act3: Tanh,
}

impl<const I: usize, const O: usize> MLPConfig<I, O> {
    pub fn new(l1: usize, l2: usize) -> Self {
        Self {
            linear1: LinearConfig::new(Const, l1),
            act1: Default::default(),
            linear2: LinearConfig::new(l1, l2),
            act2: Default::default(),
            linear3: LinearConfig::new(l2, Const),
            act3: Default::default(),
        }
    }
}

impl<const I: usize, const O: usize> MLP<I, O, f32, Device> {
    /// Save the MLP parameters to a folder of `.safetensor` files
    pub fn save_safetensors<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), SafeTensorError> {
        let mut tensors = Vec::new();

        self.linear1.write_safetensors("linear1", &mut tensors);
        self.linear2.write_safetensors("linear2", &mut tensors);
        self.linear3.write_safetensors("linear3", &mut tensors);

        let data = tensors.iter().map(|(k, dtype, shape, data)| {
            (
                k.clone(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        });

        safetensors::serialize_to_file(data, &None, path.as_ref())
    }

    /// Load the MLP parameters from a folder of `.safetensor` files
    pub fn load_safetensors<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), SafeTensorError> {
        let f = std::fs::File::open(path)?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&f)? };
        let tensors = safetensors::SafeTensors::deserialize(&buffer)?;

        self.linear1.read_safetensors("linear1", &tensors)?;
        self.linear2.read_safetensors("linear2", &tensors)?;
        self.linear3.read_safetensors("linear3", &tensors)?;

        Ok(())
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

    /// Fit the model to the input data for the specified number of epochs
    /// 
    /// This function allows for warm-start training and uses the ADAM optimisation loop.
    pub fn partial_fit(&mut self, x: Tensor<(usize, Const<R>), f32, Device>, epochs: usize, lr: f64) {
        // Initialise device
        let dev: Device = Device::default();

        // Define architecture
        let mut model = if let Some(m) = &self.model {
            m.clone()
        } else {
            let arch = (self.encoder.clone(), self.decoder.clone());
            dev.build_module::<f32>(arch)
        };
        let mut grads = model.alloc_grads();

        // Initialise optimiser
        let mut opt = Adam::new(
            &model,
            AdamConfig {
                lr,
                weight_decay: None,
                ..Default::default()
            },
        );

        // Optimisation loop
        for i in 0..epochs {
            // Collect the gradients of the network
            let prediction = model.forward_mut(x.trace(grads));
            let loss = mse_loss(prediction, x.clone());
            println!("Training loss after {i}: {:?}", loss.array());
            grads = loss.backward();

            // Update weights
            opt.update(&mut model, &grads)
                .expect("Oops, there were some unused params");

            model.zero_grads(&mut grads);
        }

        self.model = Some(model)
    }

    /// Calculate the MSE loss on a set of validation data
    pub fn calc_validation_loss(&self, x: Tensor<(usize, Const<R>), f32, Device>) -> f32 {
        let prediction = self.model.as_ref().unwrap().forward(x.clone());

        mse_loss(prediction, x).as_vec()
            .iter()
            .fold(0., |acc, i| acc + i)
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

    /// Save the model parameters to a pair of `.safetensor` files
    pub fn save(&self, path: &Path, model_name: &str) -> Result<(), SafeTensorError> {
        let mut p_encoder = path.to_path_buf();
        let mut p_decoder = path.to_path_buf();
        p_encoder.push(model_name.to_owned() + "_encoder.safetensors");
        p_decoder.push(model_name.to_owned() + "_decoder.safetensors");

        self.model
            .as_ref()
            .expect("Encoder model not initialised")
            .0
            .save_safetensors(p_encoder)?;
        self.model
            .as_ref()
            .expect("Decoder model not initialised")
            .1
            .save_safetensors(p_decoder)?;
        Ok(())
    }

    /// Load the model parameters from a pair of `.safetensor` files
    pub fn load(
        &mut self,
        encoder_path: &Path,
        decoder_path: &Path,
    ) -> Result<(), SafeTensorError> {
        match self.model.as_mut() {
            Some(m) => {
                m.0.load_safetensors(encoder_path)?;
                m.1.load_safetensors(decoder_path)?;
            }
            None => {
                let dev: Device = Device::default();
                let arch = (self.encoder.clone(), self.decoder.clone());
                let mut m = dev.build_module::<f32>(arch);

                m.0.load_safetensors(encoder_path)?;
                m.1.load_safetensors(decoder_path)?;
                self.model = Some(m);
            }
        }

        Ok(())
    }
}
