# Phoenix

Phoenix predicts survey response probabilities for SimulacraBench. It combines
smoothed conditional counts, survey skip rules, and cross-fitted TabICLv2.
For the World Bank survey, it also selects a TabPFN 3.5 Fast blend on a holdout
of visible respondents. The entry point is
[`predict(frame, schema)`](submission/main.py).

## Build

Use Python 3.12 or later. Download the pinned assets:

- [TabICLv2 checkpoint](https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt)
- [TabPFN 3.5 Fast checkpoint](https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/tabpfn-v3.5-fast-20260909.safetensors)
- [TabPFN 9.1.0 wheel](https://files.pythonhosted.org/packages/69/7d/fa98749f072dc6b88656b5f953ee86f734a90ca5c78966c8033fb2ed6c0d/tabpfn-9.1.0-py3-none-any.whl)

~~~sh
python verify_submission.py --output phoenix.zip \
  --tabicl tabicl-classifier-v2-20260212.ckpt \
  --tabpfn tabpfn-v3.5-fast-20260909.safetensors \
  --tabpfn-wheel tabpfn-9.1.0-py3-none-any.whl
~~~

The builder extracts the column embedding from the wheel and checks every
file against the submission manifest. To compare an original competition
archive byte for byte, run `python verify_submission.py --archive original.zip`.

## Run

Use the [SimulacraBench starter kit](https://github.com/SituatedEvals/public/tree/aa03186f8f083c7785a1a566c802c2077925e265)
to check the ZIP and run the Phase 2 sandbox for each survey. The full run
uses CUDA and the [submission dependencies](submission/requirements.txt).

## License

Phoenix code is released under the [MIT License](LICENSE). Vendored code
and model weights retain their upstream licenses. See
[vendor notices](submission/vendor/THIRD-PARTY-NOTICES.md) and the
[TabPFN checkpoint license](submission/LICENSE.TabPFN.weights.txt).
