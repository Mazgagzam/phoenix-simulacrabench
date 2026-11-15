# Phoenix · Test 1053693 source

The files under submission/ come from the exact competition payload
submitted as Test 1053693 and previously as Development 987695. The
program entry point is predict(frame, schema) in submission/main.py.
This source distribution omits three separately obtainable model assets.

## Verify and rebuild

From this directory with Python 3.12 or later, run:

~~~sh
python verify_submission.py --archive /path/to/submission_phoenix_pfn-wb-adaptive-tab.zip
~~~

This verifies both the archive SHA-256 and the 232 member SHA-256 hashes.
For a rebuild from source, get the pinned upstream files:

1. [TabICLv2 checkpoint](https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt).
2. [TabPFN 3.5 Fast checkpoint](https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/tabpfn-v3.5-fast-20260909.safetensors).
3. [TabPFN 9.1.0 wheel](https://files.pythonhosted.org/packages/69/7d/fa98749f072dc6b88656b5f953ee86f734a90ca5c78966c8033fb2ed6c0d/tabpfn-9.1.0-py3-none-any.whl).

~~~sh
python verify_submission.py --output rebuilt.zip --tabicl /path/to/tabicl-classifier-v2-20260212.ckpt --tabpfn /path/to/tabpfn-v3.5-fast-20260909.safetensors --tabpfn-wheel /path/to/tabpfn-9.1.0-py3-none-any.whl
~~~

The rebuild verifies the wheel and all 232 member hashes. Toolchain and
ZIP metadata can affect ZIP container bytes. Use the archived competition ZIP
when you need the original container checksum.

With the [organizer starter kit](https://github.com/SituatedEvals/public/tree/aa03186f8f083c7785a1a566c802c2077925e265),
run its ZIP checker, then use its sandbox generator and scorer with phase 2
for each schema. The full workload uses a CUDA GPU and the hosted runner's
Python 3.13 and PyTorch. The predictor requirements are in
submission/requirements.txt. The synthetic sandbox verifies execution and
output format. The official respondent splits remain with the organizers.

## Licenses and attribution

See submission/vendor/THIRD-PARTY-NOTICES.md and the vendor license files.
The TabPFN checkpoint uses the bundled Prior Labs non-commercial license
and attribution notice. Rights to use that checkpoint are granted directly
by Prior Labs under its own terms. The root MIT license applies to Phoenix-authored code. Vendored components
and pretrained model assets retain their respective upstream terms.
