import numpy as np
import cupy as cp
import itertools
import time


# setting--------------------------------
snrdB_iteration=100
snrdB_default=0
SNR_INTERVAL=1
word_error_iteration=100
order=0
n = 24  # 次元数
k = 12
G = cp.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
            ])
# setting_end--------------------------------
cp.set_printoptions(threshold=cp.inf)

def main():
    with open('data/osd.txt', 'w') as f:
        print(f"初期設定-------------------------",file=f)
        print(f"Order:{order}",file=f)
        print(f"符号長n:{n}",file=f)
        print(f"情報記号数:{k}",file=f)
        # print(f"delta:{delta}",file=f)
        # print(f"t2:{t2}",file=f)
        print(f"生成行列G\n{G}",file=f)
        # print(f"情報記号:\n{m}",file=f)
        # print(f"送信符号語:\n{x}",file=f)
        print(f"情報記号はランダム生成",file=f)
        print(f"snrdB_iteration:{snrdB_iteration}",file=f)
        print(f"snrdb_default:{snrdB_default}",file=f)
        print(f"word_error_iteration:{word_error_iteration}",file=f)
        print(f"初期設定end-----------------------",file=f)


    with open("data/osd_onlydata.txt", "w") as f2:
        print(f"初期設定-------------------------",file=f2)
        print(f"Order:{order}",file=f2)
        print(f"符号長n:{n}",file=f2)
        print(f"情報記号数:{k}",file=f2)
        # print(f"delta:{delta}",file=f2)
        # print(f"t2:{t2}",file=f2)
        print(f"生成行列G\n{G}",file=f2)
        # print(f"情報記号:\n{m}",file=f2)
        # print(f"送信符号語:\n{x}",file=f2)
        print(f"情報記号はランダム生成",file=f2)
        print(f"snrdB_iteration:{snrdB_iteration}",file=f2)
        print(f"snrdb_default:{snrdB_default}",file=f2)
        print(f"word_error_iteration:{word_error_iteration}",file=f2)
        print(f"初期設定end-----------------------",file=f2)

        pass

    #------------------------線形符号計算--------------------------
    # values=[0,1]
    # m_combinations = cp.array(list(itertools.product(values, repeat=k)))
    # candidate_codeword = codeword_create(G, m_combinations)
    # Linear_code_bpsk=cp.where(candidate_codeword==0,1,-1)
    # print(f"Linear_code_bpsk:\n{Linear_code_bpsk}")
    #--------------------------------------------------------
    #-----------------------D(組み合わせ)の計算-------------------
    # values=[-1,1]
    # mxTwoCombinations=np.array(list(itertools.product(values,repeat=2*m_range)))
    # Minus1Combinations = np.insert(mxTwoCombinations, m_range, -1, axis=1)
    # Minus1Combinations=cp.array(Minus1Combinations)
    # print(f"MinusCombinations:\n{Minus1Combinations}")
    #------------------------------------------------------------



    #------------------------Orderに従って、error位置を作る-------------------
    all_combinations=[]
    for i in range(order+1):
        arr = [0] * k  # 全て0で初期化
        arr[:i] = [1] * i  # 最初のk個を1にする
        combinations=generate_combinations(arr)
        print(f"combinations:\n{combinations}")
        all_combinations=all_combinations+combinations
    error_positions=cp.array(all_combinations)
    #---------------------------------------------------------

    

    for i in range(snrdB_iteration):
        snrdB=i*SNR_INTERVAL+snrdB_default
        t1 = create_t_1_from_snrdB(snrdB, n, k)

        #相関通信路
        # sigma = create_sigma(n, t1)
        # print(f"sigma:\n{sigma}")
        #相関通信路end

        #無相関通信路
        sigma=t1*cp.eye(n) #単位行列
        print(f"sigma:\n{sigma}")
        #無相関通信路end
            
        word_error_count = 0
        iteration_count = 0
        while word_error_count < word_error_iteration:
            time_start1 = time.time()
            m = m_create(k)
            x = codeword_create(G, m)
            r = received_sequence_create(x, sigma)

            pi1=sortPermutationMatrixByReceivedSequence(r)
            y=hardDecisionFromReceivedSequence(r)

            y2=cp.dot(y,pi1)%2
            # print(f"r:\n{cp.dot(r,pi1)}")
            # print(f"y2:\n{y2}")
            G2=cp.dot(G,pi1)%2

            G2_copy=G2.copy()

            U,pi2=gaussian_elimination(G2_copy)

            y3=cp.dot(y2,pi2)%2
            # print(f"y3:\n{y3}")
            # print(f"x:\n{cp.dot(cp.dot(x,pi1)%2,pi2)}")
            
            G3=cp.dot(cp.dot(U,G2)%2,pi2)

            u0=y3[0:k]

            x_estimate=OSD(u0,G3,r,sigma,pi1,pi2,error_positions)


            print("----------------------------------")
            print(f"snrdB:{snrdB}")
            print(f"情報記号:{m}")
            print(f"送信符号語:{x}")
            # print(f"正解の尤度:{correct_loglikehood}")
            # print(f"G:\n{G}")

            # max_loglikehood, x_estimate=batch_mle_calculate(G,k,r,sigma,limited_memory)
            # max_loglikehood, x_estimate=pre_batch_max_loglikehood_estimate_calculate(G, k, r, sigma)
            # print(f"v_bpsk:\n{v_bpsk}")
            # print(f"estimate_v:\n{x_estimate}")

            #--------WordErrorRate-----------
            if cp.all(x == x_estimate):
                print(f"復号成功")
            else:
                print(f"復号失敗")
                word_error_count +=1
            #---------------------------------


            time_end1 = time.time()
            # print(f"出力尤度:{max_loglikehood}")
            print(f"推定符号語:{x_estimate}")
            print(f"現在までの復号誤り個数:{word_error_count}")
            # print(f"現在までの復号bit誤り個数:{bit_error_count}")
            print(f"現在までの反復回数:{iteration_count+1}")
            print(f"時間:{time_end1-time_start1}")

            iteration_count +=1

        print(f"wer:{word_error_count/iteration_count}")
        # print(f"ber:{bit_error_count/(iteration_count*n)}")
        with open("data/osd.txt","a") as f:
            print(f"------------------------------",file=f)
            print(f"snrdB:{snrdB}",file=f)
            print(f"{iteration_count}回目,合計誤り回数:{word_error_count}",file=f)
            # print(f"{iteration_count}回目,合計bit誤り回数:{bit_error_count}",file=f)
            print(f"WER:{word_error_count/iteration_count}",file=f)
            # print(f"BER:{bit_error_count/(iteration_count*n)}",file=f)
            # print(f"Minute_Value_n:\n{Minute_Value_n}",file=f)
        with open("data/osd_onlydata.txt","a")as f2:
            print(f"{word_error_count/iteration_count},",file=f2)
            # print(f"{bit_error_count/(iteration_count*n)},",file=f2)

    # ------------------------------------------

    return 0

def generator_matrix_random(n, k):
    G = cp.random.randint(0, 2, size=(k,n))
    return G


def m_create(k):
    return cp.random.randint(0, 2, size=(k,))


def codeword_create(G, m):
    return cp.dot(m, G) % 2


def create_t_1_from_snrdB(snrdB, n, k):
    Eb = n/k  # =1/R
    N0 = Eb/(cp.power(10, snrdB/10))
    t1 = N0/2  # 分散はN0/2
    print(f"Eb:{Eb}")
    print(f"N0:{N0}")
    print(f"t1(分散):{t1}")
    return t1


def noise_create(sigma):
    n = sigma.shape[0]
    mu = cp.zeros(n)
    noises = cp.random.multivariate_normal(mu, sigma, method='cholesky',dtype=np.float64,check_valid='raise',tol=1e-08)

    return noises


def received_sequence_create(x, sigma):
    x_bpsk=cp.where(x==0,1,-1)
    z = noise_create(sigma)
    r = x_bpsk+z

    # print(f"v:\n{v}")
    # print(f"noises:\n{z}")
    # print(f"r:\n{r}")
    return r


# def k_ij(i, j, t1):  # カーネル

    return t1*cp.exp((-cp.power(delta*i-delta*j, 2))/t2)


# def create_sigma(n, t1):
    sigma = cp.empty((n, n))
    for i in range(n):
        for j in range(n):
            sigma[i, j] = k_ij(i, j, t1)
    return sigma


def multi_logpdf(x, means, cov):
    L = cp.linalg.cholesky(cov)
    k=means.shape[0]
    n=means.shape[1]
    dev = x-means  # (k x n)
    dev=dev.reshape(k,n,1) 
    # Lの逆行列を求め、devを変換
    # z = np.linalg.solve(L.T, dev.T).T  # L^Tに対する線形方程式を解く
    L_expanded = cp.tile(L, (k, 1, 1))
    # print(f"dev_reshape:\n{dev.shape}")
    # print(f"L:\n{L}")
    # print(f"L_expand:\n{L_expanded}")
    z = cp.linalg.solve(L_expanded, dev)
    maha=z.transpose(0, 2, 1)@z
    # result=cp.exp(-0.5*maha)/((2*cp.pi)**n)*cp.linalg.det(cov)
    result=cp.exp(-0.5*maha)
    
    return result


def MahalanobisDistancebyCholesky(ys,xs,cov):
    L = cp.linalg.cholesky(cov)
    k=xs.shape[0]
    n=xs.shape[1]
    devs = ys-xs  # (k x n)
    devs=devs.reshape(k,n,1) 
    L_expanded = cp.tile(L, (k, 1, 1))
    z = cp.linalg.solve(L_expanded, devs)
    maha=z.transpose(0, 2, 1)@z

    return maha

def sortPermutationMatrixByReceivedSequence(r):
    n=r.shape[0]
    identity_matrix=cp.eye(n)
    sorted_indices=cp.argsort(-cp.abs(r))
    permutation_matrix=identity_matrix[: , sorted_indices]
    # print(f"sorted_indices:\n{sorted_indices}")
    # print(f"ソートr:\n{r[sorted_indices]}")
    # print(f"permutatin:\n{permutation_matrix}")

    return permutation_matrix

def hardDecisionFromReceivedSequence(r):
    hard_decision = cp.where(r >= 0, 0, 1)
    
    return hard_decision

def gaussian_elimination(A):
    number_of_row=A.shape[0]
    number_of_column=A.shape[1]
    U=cp.eye(number_of_row)
    P=cp.eye(number_of_column) 

    i=0
    j=0
    while i!=number_of_row:
        if (A[i][i]==1):
            #前進消去（ピボットから同列の非ゼロ成分を消す）
            indices = cp.where((A[:, i] == 1) & (cp.arange(A.shape[0]) != i))[0]
            A[indices] = (A[indices] + A[i])%2
            U[indices] = (U[indices] + U[i])%2
            i+=1
        else: #A[i][i]==0 ピボットが零である
            non_zero_position_except_for_iielement = cp.where(A[i+1:, i] == 1)[0]
            if non_zero_position_except_for_iielement.size > 0:
                position=int(non_zero_position_except_for_iielement[0] + (i + 1))
                A[[i, position ],:] = A[[position, i],:]
                U[[i, position ],:] = U[[position, i],:] 
            else:
                j+=1
                A[:,[i,i+j]]=A[:,[i+j,i]]
                P[:,[i,i+j]]=P[:,[i+j,i]]
                # print(f"j:\n{j}")

        # print(f"A:\n{A}")
        # print(f"U:\n{U}")
        # print(f"P:\n{P}")
        # print(f"i:\n{i}")
    

    # print(f"A:\n{A}")
    return U,P


def generate_combinations(array):
    n = len(array)
    ones_count = array.count(1)
    return [
        tuple(1 if i in comb else 0 for i in range(n))
        for comb in itertools.combinations(range(n), ones_count)
    ]

def OSD(u0,G,r,sigma,pi1,pi2,comb):
    u0s=(comb+u0)%2
    candidate_codewords=codeword_create(G,u0s)


    candidate_codewords_pi1_pi2=cp.dot(cp.dot(candidate_codewords,pi2.T),pi1.T)
    
    # print(f"pi2:\n{pi2}")
    # print(f"candidate_codeword:\n{candidate_codewords}")
    # print(f"dot(pi2,cand):\n{cp.dot(candidate_codewords,pi2.T)}")
    # print(f"pi1:\n{pi1}")
    # print(f"candidate_codewords_pi1_pi2:\n{candidate_codewords_pi1_pi2}")
    candidate_codewords_pi1_pi2_bpsk=cp.where(candidate_codewords_pi1_pi2==0,1,-1)
    loglikehoods=MahalanobisDistancebyCholesky(r,candidate_codewords_pi1_pi2_bpsk,sigma)

    # loglikehoods=new_multi_logpdf(r,candidate_codewords_pi1_pi2_bpsk,sigma)
    # loglikehoods=MahalanobisDistances(r,candidate_codewords_pi1_pi2_bpsk,sigma)
    # loglikehoods=OnlyEuclideanDistance(r,candidate_codewords_pi1_pi2_bpsk,t1)
    # print(f"loglikehoods:\n{loglikehoods}")
    # print(f"r:\n{r}")
    # print(f"候補符号語:\n{candidate_codewords_pi1_pi2}")
    # print(f"尤度\n{loglikehoods}")
    # max_loglikehood=cp.min(loglikehoods)
    # max_loglikehood_index=cp.argmin(loglikehoods)
    # print(f"loglikehoods.shape:\n{loglikehoods}")
    max_loglikehood=cp.min(loglikehoods)
    max_loglikehood_index=cp.argmin(loglikehoods)
    # print(f"max_loglikehood:\n{max_loglikehood}")
    # print(f"max_loglikehood_index:\n{max_loglikehood_index}")
    codeword_estimate=candidate_codewords_pi1_pi2[max_loglikehood_index]
    if cp.isnan(max_loglikehood):
        raise ValueError("エラーが発生しました:尤度がすべてnan")
    # print(f"u0:\n{u0}")
    # print(f"u0s:\n{u0s}")
    # print(f"candidate_codewords:\n{candidate_codewords}")
    # print(f"codeword_estimate:\n{codeword_estimate}")
    return codeword_estimate
        

if __name__ == "__main__":
    main()