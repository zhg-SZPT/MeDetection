import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def coore1(name,N):
    corrects=np.load(name)
    # 每N个取一个点
    finally_corrects = []
    for i in range(0, len(corrects), N):
        try:
            finally_corrects.append(corrects[i])
        except:
            pass
    y1 = finally_corrects
    x1 = [(i + 1)*N for i in range(0, len(y1))]
    return x1,y1
def coore2(taskname,flag):
    c =  np.load("./mynpy/" + taskname + "/"+ flag+  "-c.npy")[800:]
    r = np.load("./mynpy/" + taskname + "/" + flag + "-r.npy")[800:]
    p = np.load("./mynpy/" + taskname + "/" + flag + "-p.npy")[800:]
    f = np.load("./mynpy/" + taskname + "/" + flag + "-f.npy")[800:]
    a = np.load("./mynpy/" + taskname + "/" + flag + "-a.npy")[800:]
    idx=np.argmax(a)
    return a[idx]
    return c[idx],r[idx],p[idx],f[idx],a[idx]
tasknames = {"电缆": "Cable"}
for taskname, title in tasknames.items():
    x1, y1 = coore1("./mynpy/" + taskname + "/random-a.npy", 1)
    x2, y2 = coore1("./mynpy/" + taskname + "/maml-a.npy", 1)
    font_size = 24
    pdf = PdfPages("./res/" + taskname + ".pdf")
    figure = plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=30)
    plt.plot(x1, y1, 'r', label="pre")
    plt.plot(x2, y2, 'g', label="maml")
    plt.xlabel("epoch", fontsize=font_size)
    plt.ylabel("Recall", fontsize=font_size)
    plt.legend(loc="lower right", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    pdf.savefig()
    plt.close()
    pdf.close()