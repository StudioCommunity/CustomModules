
# python -m test.builtin_models_test
if __name__ == '__main__':
    # keras test
    from builtin_models.keras import *
    print('---keras test---')
    model = load_model_from_local_file('D:/GIT/CustomModules-migu-NewYamlTest2/dstest/model/keras-mnist/model.h5')
    print('------')
    save_model(model, "./test/outputModels/keras/")
    print('********')

    #sklearn test
    from builtin_models.sklearn import *
    print('---sklearn test---')
    model = load_model_from_local_file('D:/GIT/CustomModules-migu-NewYamlTest2/dstest/dstest/sklearn/model/sklearn/model.pkl')
    print('------')
    save_model(model, "./test/outputModels/sklearn/")
    print('********')

    #pytorch test
    from builtin_models.pytorch import *
    print('---pytorch test---')
    model = load_model_from_local_file('D:/GIT/CustomModules-migu-NewYamlTest2/dstest/model/pytorch-mnist/model.pkl')
    print('------')
    save_model(model, "./test/outputModels/pytorch/")
    print('********')