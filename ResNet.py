import keras
from keras import Model
from keras import layers as ly

#残差块
class ResidualBlock(Model):
    def __init__(self, filters, strides, use_conv):
        super(ResidualBlock, self).__init__()
        self.c1 = ly.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')
        self.b1 = ly.BatchNormalization()
        self.r1 = ly.Activation('relu')
        self.c2 = ly.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.b2 = ly.BatchNormalization()
        # 如果需要变换维度，则侧链增加1×1卷积层
        if use_conv:
            self.c3 = ly.Conv2D(filters=filters, kernel_size=1, strides=strides)
        else:
            self.c3 = None
        self.r2 = ly.Activation('relu')

    def call(self, x):
        y = self.c1(x)
        y = self.b1(y)
        y = self.r1(y)
        y = self.c2(y)
        y = self.b2(y)
        if self.c3:
            x = self.c3(x)
        out = self.r2(y + x)

        return out

#残差（块）层，每层2个残差块，网络中共四层残差块[2,2,2,2]
class ResnetBlock(ly.Layer):
    def __init__(self, filters, num_residuals, first_block=False):
        super(ResnetBlock, self).__init__()
        self.listLayers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                residual = ResidualBlock(filters=filters, use_conv=True, strides=2)
                self.listLayers.append(residual)
            else:
                residual = ResidualBlock(filters=filters, use_conv=False, strides=1)
                self.listLayers.append(residual)

    def call(self, x):
        for layer in self.listLayers:
            x = layer(x)
        return x

#batchnormalize代替dropout
class ResNet(Model):
    def __init__(self, units):
        super(ResNet, self).__init__()
        self.c1 = ly.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.b1 = ly.BatchNormalization()
        self.r1 = ly.Activation('relu')
        self.p1 = ly.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.rb1 = ResnetBlock(filters=64, num_residuals=2, first_block=True)
        self.rb2 = ResnetBlock(filters=128, num_residuals=2)
        self.rb3 = ResnetBlock(filters=256, num_residuals=2)
        self.rb4 = ResnetBlock(filters=256, num_residuals=2)
        self.ap = ly.GlobalAvgPool2D()
        self.f = ly.Dense(units, activation='relu')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.r1(x)
        x = self.p1(x)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.ap(x)
        x = self.f(x)

        return x