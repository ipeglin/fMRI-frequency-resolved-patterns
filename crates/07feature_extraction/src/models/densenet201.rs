use tch::{nn, nn::ModuleT, Tensor};

const GROWTH_RATE: i64 = 32;
const BN_SIZE: i64 = 4; // bottleneck multiplier: inter_channels = BN_SIZE * GROWTH_RATE = 128

/// DenseNet-201 architecture.
/// Huang et al., "Densely Connected Convolutional Networks", 2016.
/// https://arxiv.org/abs/1608.06993
///
/// Use `forward_features` to extract the 1920-dim feature vector (for KNN/SVM).
/// Use `forward_t` (via ModuleT) for full classification.
#[derive(Debug)]
pub struct DenseNet201 {
    features: nn::SequentialT,
    classifier: nn::Linear,
}

impl DenseNet201 {
    pub fn new(vs: &nn::Path, num_classes: i64) -> DenseNet201 {
        let fp = vs / "features";

        // Initial Convolution
        let mut features = nn::seq_t()
            .add(nn::conv2d(&fp / "conv0", 3, 64, 7, nn::ConvConfig { stride: 2, padding: 3, bias: false, ..Default::default() }))
            .add(nn::batch_norm2d(&fp / "norm0", 64, Default::default()))
            .add_fn(|x| x.relu().max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false));

        // Dense Blocks and Transition Layers
        //   Block 1:  6 layers,  64 channels in -> 256 out
        //   Trans 1:             256 -> 128
        //   Block 2: 12 layers, 128 channels in -> 512 out
        //   Trans 2:             512 -> 256
        //   Block 3: 48 layers, 256 channels in -> 1792 out
        //   Trans 3:            1792 -> 896
        //   Block 4: 32 layers, 896 channels in -> 1920 out
        features = features
            .add(Self::dense_block(&fp / "denseblock1",  6,  64))
            .add(Self::transition(&fp / "transition1", 256, 128))
            .add(Self::dense_block(&fp / "denseblock2", 12, 128))
            .add(Self::transition(&fp / "transition2", 512, 256))
            .add(Self::dense_block(&fp / "denseblock3", 48, 256))
            .add(Self::transition(&fp / "transition3", 1792, 896))
            .add(Self::dense_block(&fp / "denseblock4", 32, 896));

        // Final normalization and global average pool -> 1920-dim flat vector
        features = features
            .add(nn::batch_norm2d(&fp / "norm5", 1920, Default::default()))
            .add_fn(|x| x.relu().avg_pool2d(&[7, 7], &[1, 1], &[0, 0], false, true, 1).flat_view());

        let classifier = nn::linear(vs / "classifier", 1920, num_classes, Default::default());

        DenseNet201 { features, classifier }
    }

    /// Extract the 1920-dim feature vector before the classifier.
    /// Use this output as input to a KNN or SVM.
    pub fn forward_features(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply_t(&self.features, train)
    }

    fn dense_block(vs: nn::Path, num_layers: i64, in_channels: i64) -> nn::SequentialT {
        let mut block = nn::seq_t();
        for i in 0..num_layers {
            block = block.add(Self::dense_layer(
                &vs / &format!("denselayer{}", 1 + i),
                in_channels + i * GROWTH_RATE,
            ));
        }
        block
    }

    /// Bottleneck dense layer: BN -> ReLU -> Conv1x1 -> BN -> ReLU -> Conv3x3.
    /// Output is concatenated with input along the channel dimension (feature reuse).
    fn dense_layer(vs: nn::Path, in_channels: i64) -> impl ModuleT + 'static {
        let inter_channels = BN_SIZE * GROWTH_RATE; // 128
        let bn1   = nn::batch_norm2d(&vs / "norm1", in_channels, Default::default());
        let conv1 = nn::conv2d(&vs / "conv1", in_channels, inter_channels, 1, nn::ConvConfig { bias: false, ..Default::default() });
        let bn2   = nn::batch_norm2d(&vs / "norm2", inter_channels, Default::default());
        let conv2 = nn::conv2d(&vs / "conv2", inter_channels, GROWTH_RATE, 3, nn::ConvConfig { padding: 1, bias: false, ..Default::default() });
        nn::func_t(move |x, train| {
            let y = x.apply_t(&bn1, train).relu().apply(&conv1)
                     .apply_t(&bn2, train).relu().apply(&conv2);
            Tensor::cat(&[x, &y], 1)
        })
    }

    fn transition(vs: nn::Path, in_channels: i64, out_channels: i64) -> nn::SequentialT {
        nn::seq_t()
            .add(nn::batch_norm2d(&vs / "norm", in_channels, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(&vs / "conv", in_channels, out_channels, 1, nn::ConvConfig { bias: false, ..Default::default() }))
            .add_fn(|x| x.avg_pool2d_default(2))
    }
}

/// Full forward pass through features + classifier. Returns class logits.
impl nn::ModuleT for DenseNet201 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_features(xs, train).apply(&self.classifier)
    }
}
