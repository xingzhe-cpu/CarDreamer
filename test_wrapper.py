# 1️⃣ 定义装饰器类
class Wrapper:
    def __init__(self, cls):
        self._cls = cls
        # # cls 就是被装饰的原始类对象
        # print("[Wrapper.__init__] 收到类:", cls.__name__)
        # self._cls = cls

    def __call__(self, *args, **kwargs):
        print("[Wrapper] 创建实例")
        return self._cls(*args, **kwargs)
        # print("[Wrapper.__call__] 现在开始实例化", self._cls.__name__)
        # # 在这里可以对参数做修改、记录日志等
        # obj = self._cls(*args, **kwargs)   # 创建原始类的实例
        # # 可以给实例增加额外属性或功能
        # obj.extra = "由装饰器添加"
        # return obj

# 2️⃣ 使用类装饰器
@Wrapper
class Foo:
    def __init__(self, x):
        print("[Foo.__init__] 被调用")
        self.x = x
    def __call__(self, y):
        print(f"[Foo] __call__ 被调用, y={y}")
        return self.x + y

# 3️⃣ 测试
if __name__ == "__main__":
    f2 = Foo._cls(10)   # 直接用保存的原始类创建实例
    f2(5)               # 调用 Foo.__call__
