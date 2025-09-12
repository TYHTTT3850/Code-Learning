import numpy as np

print("===== 旧接口 (np.random) =====")
print("均匀分布 [0,1):", np.random.rand(3))
print("标准正态分布 N(0,1):", np.random.randn(3))
print("整数均匀分布 [0,10):", np.random.randint(0, 10, 5))
print("正态分布 N(5, 2^2):", np.random.normal(loc=5, scale=2, size=3))
print("均匀分布 U(-1,1):", np.random.uniform(-1, 1, 3))
print("泊松分布 λ=3:", np.random.poisson(lam=3, size=5))
print("二项分布 n=10,p=0.3:", np.random.binomial(10, 0.3, 5))
print("多项分布 n=10, p=[0.2,0.5,0.3]:", np.random.multinomial(10, [0.2,0.5,0.3]))
print("Beta 分布 a=2,b=5:", np.random.beta(2, 5, 3))
print("Gamma 分布 k=2, θ=2:", np.random.gamma(2, 2, 3))
print("卡方分布 df=4:", np.random.chisquare(4, 3))
print("F 分布 df1=3,df2=5:", np.random.f(3, 5, 3))
print("指数分布 scale=2:", np.random.exponential(2, 3))
print("几何分布 p=0.3:", np.random.geometric(0.3, 5))
print("超几何分布:", np.random.hypergeometric(10, 5, 3, 5))
print("对数正态分布 mean=0,sigma=1:", np.random.lognormal(0, 1, 3))
print("拉普拉斯分布 loc=0,scale=1:", np.random.laplace(0, 1, 3))
print("三角分布 left=0,mode=2,right=5:", np.random.triangular(0, 2, 5, 3))
print("冯·米塞斯分布 mu=0,kappa=4:", np.random.vonmises(0, 4, 3))
print("Wald (逆高斯) 分布 mean=2,scale=1:", np.random.wald(2, 1, 3))
print("Weibull 分布 a=1.5:", np.random.weibull(1.5, 3))
print("Pareto 分布 a=3:", np.random.pareto(3, 3))
print("Power 分布 a=5:", np.random.power(5, 3))

# 其他功能
arr = np.arange(10)
np.random.shuffle(arr)  # 原地打乱
print("打乱数组 shuffle:", arr)
print("打乱数组 permutation:", np.random.permutation(10))
print("从数组中采样 choice:", np.random.choice([1, 2, 3, 4, 5], size=3, replace=False, p=None))

np.random.seed(123)  # 设置随机种子
print("设置随机种子后的随机数:", np.random.rand(3))

print("\n===== 新接口(Generator) =====")
rng = np.random.default_rng(seed=42)

print("均匀分布 [0,1):", rng.random(3))
print("整数均匀分布 [0,10):", rng.integers(0, 10, 5))
print("标准正态分布 N(0,1):", rng.standard_normal(3))
print("正态分布 N(5,2^2):", rng.normal(5, 2, 3))
print("均匀分布 U(-1,1):", rng.uniform(-1, 1, 3))
print("泊松分布 λ=3:", rng.poisson(3, 5))
print("二项分布 n=10,p=0.3:", rng.binomial(10, 0.3, 5))
print("多项分布 n=10, p=[0.2,0.5,0.3]:", rng.multinomial(10, [0.2,0.5,0.3]))
print("Beta 分布 a=2,b=5:", rng.beta(2, 5, 3))
print("Gamma 分布 k=2,θ=2:", rng.gamma(2, 2, 3))
print("卡方分布 df=4:", rng.chisquare(4, 3))
print("F 分布 df1=3,df2=5:", rng.f(3, 5, 3))
print("指数分布 scale=2:", rng.exponential(2, 3))
print("几何分布 p=0.3:", rng.geometric(0.3, 5))
print("超几何分布 (ngood=10,nbad=5,nsample=3):", rng.hypergeometric(10, 5, 3, 5))
print("对数正态分布 mean=0,sigma=1:", rng.lognormal(0, 1, 3))
print("拉普拉斯分布 loc=0,scale=1:", rng.laplace(0, 1, 3))
print("三角分布 left=0,mode=2,right=5:", rng.triangular(0, 2, 5, 3))
print("冯·米塞斯分布 mu=0,kappa=4:", rng.vonmises(0, 4, 3))
print("Wald (逆高斯) 分布 mean=2,scale=1:", rng.wald(2, 1, 3))
print("Weibull 分布 a=1.5:", rng.weibull(1.5, 3))
print("Pareto 分布 a=3:", rng.pareto(3, 3))
print("Power 分布 a=5:", rng.power(5, 3))

# 其他功能
arr2 = np.arange(10)
rng.shuffle(arr2)  # 原地打乱
print("打乱数组 shuffle:", arr2)
print("打乱数组 permutation:", rng.permutation(10))
print("从数组中采样 choice:", rng.choice([1, 2, 3, 4, 5], size=3, replace=False, p=None))

rng2 = np.random.default_rng(seed=123)  # 新接口的种子设定
print("设置随机种子后的随机数:", rng2.random(3))