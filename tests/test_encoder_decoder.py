from fractal_brain import FusionModel, NativeEncoder, NativeAutoregressiveDecoder, set_seed, Vector


def test_fusion_model_generate_sequence() -> None:
    set_seed(3)
    encoder = NativeEncoder(vocab_size=32, d_model=8)
    decoder = NativeAutoregressiveDecoder(vocab_size=32, d_model=8)
    model = FusionModel(encoder, decoder)
    out = model.generate([1, 2, 3], max_new_tokens=5)
    assert len(out) == 5
    assert all(isinstance(i, int) for i in out)


def test_encoder_returns_latent_vector() -> None:
    set_seed(4)
    encoder = NativeEncoder(vocab_size=16, d_model=6)
    latent = encoder.encode([1, 2, 3])
    assert isinstance(latent, Vector)
    assert len(latent) == 6
