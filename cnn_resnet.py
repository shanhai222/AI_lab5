import numpy as np
import pandas as pd
import os
import cv2
from keras.models import Model, load_model
from keras.layers import Input, Conv1D, Dropout, Dense, GlobalMaxPooling1D, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import ResNet

def convert_emo_to_int(string):
    num = 0
    if string == 'positive':
        num = 0
    elif string == 'neutral':
        num = 1
    elif string == 'negative':
        num = 2

    return num

# 4000条训练数据, 511条测试数据
def load_data(isTrain):
    # 获取标签
    label = pd.read_csv("./dataset/train.txt", sep=',')

    folder = './dataset/data'
    path_list = os.listdir(folder)
    path_list.sort(key=lambda x: int(x.split('.')[0]))  # 排序

    txt_list = []
    img_dict = {}
    for filename in path_list:
        # 获取guid和对应的情绪
        id = int(os.path.splitext(filename)[0])
        emo_query = np.array(label.query("guid == @id")).tolist()
        # 训练和测试阶段读取不同的数据
        if isTrain:
            if len(emo_query) == 0:
                continue
            emo = convert_emo_to_int(emo_query[0][1])
            # txt文件
            if os.path.splitext(filename)[1] == ".txt":
                path = os.path.join(folder, filename)
                # utf-8和gb18030都有
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        line = f.readline().strip()
                    except:
                        with open(path, 'r', encoding='gb18030') as f:
                            line = f.readline().strip()
                txt_list.append([id, line, emo])

            # jpg文件
            elif os.path.splitext(filename)[1] == ".jpg":
                path = os.path.join(folder, filename)
                img = cv2.imread(path)
                # 图片尺寸不一样
                img_norm = cv2.resize(img, dsize=(224, 224))
                img_dict[id] = img_norm
        else:
            if len(emo_query) != 0:
                continue
            # txt文件
            if os.path.splitext(filename)[1] == ".txt":
                path = os.path.join(folder, filename)
                # utf-8和gb18030都有
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        line = f.readline().strip()
                    except:
                        with open(path, 'r', encoding='gb18030') as f:
                            line = f.readline().strip()
                txt_list.append([id, line])

            # jpg文件
            elif os.path.splitext(filename)[1] == ".jpg":
                path = os.path.join(folder, filename)
                img = cv2.imread(path)
                # 图片尺寸不一样
                img_norm = cv2.resize(img, dsize=(224, 224))
                img_dict[id] = img_norm

    if isTrain:
        txt = pd.DataFrame(txt_list, columns=['guid', 'text', 'label'])
    else:
        txt = pd.DataFrame(txt_list, columns=['guid', 'text'])

    return txt, img_dict

def multimodal(units):
    # 提取文本特征，TextCNN，两层卷积
    text_input = Input(shape=(1, 1000))
    conv1 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(text_input)
    drop1 = Dropout(0.5)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(drop1)
    pool1 = GlobalMaxPooling1D()(conv2)
    drop2 = Dropout(0.5)(pool1)
    text_features = Dense(units, activation='relu')(drop2)

    # 提取图片特征，ResNet
    image_input = Input(shape=(224, 224, 3))
    resnet_model = ResNet.ResNet(units)
    image_features = resnet_model(image_input)

    # 融合特征
    merged_features = concatenate([text_features, image_features])

    # 全连接层实现分类
    output = Dense(3, activation='softmax')(merged_features)

    model = Model(inputs=[text_input, image_input], outputs=output)

    return model

def calculate_acc(list1, list2):
    correct = 0
    length = len(list1)
    for i in range(length):
        if list1[i] == list2[i]:
            correct += 1

    return round(correct/length, 4)

if __name__ == "__main__":
    txt_train, img_train = load_data(True)
    txt_test, img_test = load_data(False)

    # 划分训练集和验证集(df)
    train = txt_train.sample(frac=0.8)
    val = txt_train[~txt_train.index.isin(train.index)]
    id_train = train['guid'].to_numpy()
    id_val = val['guid'].to_numpy()

    x_txt_train = train['text'].to_numpy()
    x_img_train = []
    y_train = train['label'].to_numpy()
    x_txt_val = val['text'].to_numpy()
    x_img_val = []
    y_val = val['label'].to_numpy()
    for id in id_train:
        x_img_train.append(img_train[id])
    for id in id_val:
        x_img_val.append(img_train[id])
    # list->np.array
    x_img_train = np.array(x_img_train)
    x_img_val = np.array(x_img_val)

    # tf-idf将文本映射成向量
    tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=1000)
    tfidf.fit(txt_train['text'].to_numpy())
    x_txt_train = tfidf.transform(x_txt_train)
    x_txt_val = tfidf.transform(x_txt_val)
    x_txt_train = np.expand_dims(x_txt_train.toarray(), axis=1)
    x_txt_val = np.expand_dims(x_txt_val.toarray(), axis=1)

    # # 调参
    # # 定义超参数空间
    # param_lr = [0.001, 0.005]
    # param_units = [64, 128, 256]
    # epochs = 10
    # # 颜色映射
    # colormap = plt.cm.get_cmap('viridis', 6)
    #
    # best_score = None
    # best_params = {}
    # i = 0
    # # 循环遍历超参数组合
    # for lr in param_lr:
    #     for units in param_units:
    #         color = colormap(i)
    #         # 根据当前超参数组合构建新的模型
    #         model = multimodal(units)
    #         model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #
    #         # 训练新的模型
    #         history = model.fit([x_txt_train, x_img_train], y_train, epochs=epochs, batch_size=32)
    #         N = np.arange(0, epochs)
    #         plt.plot(N, history.history.get('loss'), color=color, label='lr{}_units{}'.format(str(lr), str(units)))
    #         plt.plot(N, history.history.get('accuracy'), color=color)
    #         i += 1
    #         # 在验证集上评估新模型的性能
    #         score = model.evaluate([x_txt_val, x_img_val], y_val)
    #         acc_score = score[1]
    #
    #         # 更新最佳得分和最佳参数
    #         if best_score is None or acc_score > best_score:
    #             best_score = acc_score
    #             best_params = [lr, units]
    #
    # print("Best Accuracy in Validation set:", best_score)
    # print("Best Parameters: learning rate " + str(best_params[0]) + ", " "units " + str(best_params[1]))
    #
    # plt.legend()  # 添加图例
    # plt.xlabel('epochs')
    # plt.ylabel('loss/acc')
    # plt.title('loss/acc Change with different params')
    # plt.savefig('cnn_resnet_params.png')
    # plt.show()

    # 训练模型
    model = multimodal(128)
    model.summary()
    model.compile(optimizer=Adam(0.005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 早停防止过拟合
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode='min')
    history = model.fit([x_txt_train, x_img_train], y_train, callbacks=[earlyStop], epochs=10, batch_size=32, validation_data=([x_txt_val, x_img_val], y_val))
    # history = model.fit([x_txt_train, x_img_train], y_train, epochs=10, batch_size=32, validation_data=([x_txt_val, x_img_val], y_val))
    #model.save('cnn_resnet.h5')
    current_epoch = earlyStop.stopped_epoch + 1

    # 绘制训练曲线
    plt.figure()
    N = np.arange(0, current_epoch)
    plt.plot(N, history.history.get('loss'), color='r', label='train_loss')
    plt.plot(N, history.history.get('accuracy'), color='g', label='train_acc')
    plt.plot(N, history.history.get('val_loss'), color='orange', label='val_loss')
    plt.plot(N, history.history.get('val_accuracy'), color='b', label='val_acc')

    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss/acc')
    plt.title('loss/acc Change with epochs in Mnist')

    plt.savefig('cnn_resnet.png')
    plt.show()

    # 预测
    # model = load_model('cnn_resnet.h5')
    x_txt_test = txt_test['text'].to_numpy()
    x_img_test = []
    id_test = txt_test['guid'].to_numpy()
    for id in id_test:
        x_img_test.append(img_test[id])
    # list->np.array
    x_img_test = np.array(x_img_test)

    tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=1000)
    tfidf.fit(x_txt_test)
    x_txt_test = tfidf.transform(x_txt_test)
    x_txt_test = np.expand_dims(x_txt_test.toarray(), axis=1)
    pred = model.predict([x_txt_test, x_img_test])
    pred = np.argmax(pred, axis=1)
    test_id = txt_test['guid'].to_numpy()

    with open('./pred.txt', 'a', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for i in range(len(test_id)):
            pred_i = pred[i]
            id = test_id[i]
            f.write(str(id) + ',' + str(pred_i) + '\n')

    # 消融实验1, 用0覆盖另一模态的数据
    #model = load_model('cnn_resnet.h5')
    x_txt_abla = np.zeros((800, 1, 1000))
    pred_txt_abla = model.predict([x_txt_abla, x_img_val])
    pred_txt_abla = np.argmax(pred_txt_abla, axis=1)
    acc = calculate_acc(y_val, pred_txt_abla)
    print("The accuracy of ablation experiment one: " + str(acc))

    # 消融实验2
    #model = load_model('cnn_resnet.h5')
    x_img_abla = np.zeros((800, 224, 224, 3))
    pred_img_abla = model.predict([x_txt_val, x_img_abla])
    pred_img_abla = np.argmax(pred_img_abla, axis=1)
    acc = calculate_acc(y_val, pred_img_abla)
    print("The accuracy of ablation experiment two: " + str(acc))