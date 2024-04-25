from prepare_spectra import prepare_spectra


params = {
    "preparespectra": {
        "inputspectrum": "/home/pcorchoc/Develop/HBSPS/test/BaseGM_mock_spectra.dat",
    },
    "othermodule": {"AnotherParam": 2},
}
print(params)
prepare_spectra.setup(params)
prepare_spectra.execute()
