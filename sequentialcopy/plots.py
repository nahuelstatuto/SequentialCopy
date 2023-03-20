import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

path = '/home/nahuel.statuto/fonts/Mabry Pro/'
mabry_light = os.path.join(path, 'Mabry Pro Light.otf')
mabry_normal = os.path.join(path, 'Mabry Pro.otf')
mabry_italic = os.path.join(path, 'Mabry Pro Italic.otf')
mabry_bold = os.path.join(path, 'Mabry Pro Bold.otf')

path = '/home/nahuel.statuto/fonts/otf/'
esade_light = os.path.join(path, 'Esade-Light.otf')
esade_normal = os.path.join(path, 'Esade-Regular.otf')
esade_bold = os.path.join(path, 'Esade-Bold.otf')

font_manager.fontManager.addfont(mabry_light)
prop = font_manager.FontProperties(fname=mabry_light, size=12)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

c_online = '#1984c5'
c_seq = '#a12e21'
colors = ["#e1a692", "#de6e56", "#e14b31", "#a12e21", "#1984c5"]

def plot_results(model, plot_type='acc_test'):
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Iteration')
    x = np.arange(1, len(model.acc_test)+1)
    ax.set_xticks(np.arange(1, len(model.acc_test)+1))

    if plot_type == 'acc_test':
        plt.plot(x, model.acc_test,'o--')
        ax.set_ylabel('Acc test')
    elif plot_type == 'nN':
        plt.plot(x, model.n,'o--')
        ax.set_ylabel('Number of synthetic data points')
    elif plot_type == 'rho':
        plt.plot(x, model.rho,'o--')
        ax.set_ylabel('Average uncertainty rho')
    else:
        pass

    plt.show()

def iteration_plot(data, 
                   labels,
                   max_iter = 3,
                   colors = colors,
                   fill=True,
                   ylabel='Accuracy',
                   single_pass=False,
                   one_shot=None,
                   path=None):
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    
    for i, label in enumerate(labels):
        ax.plot(np.arange(1, max_iter+1), data[i].mean(axis=0), '-o',label=label, c=colors[-(i+1)] if single_pass else colors[-(i+2)])
        if fill:
            ax.fill_between(np.arange(1, max_iter+1), data[i].mean(axis=0)-data[i].std(axis=0), data[i].mean(axis=0)+data[i].std(axis=0), alpha=0.2, color=colors[-(i+1)] if single_pass else colors[-(i+2)])
            
    one_shot_baseline(ax, one_shot, max_iter)

    ax.set_xlabel('Iterations')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.85,max_iter+0.15)
    ax.set_xticks(np.arange(1, max_iter+1))
    
    set_tick_params(ax)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    
    bbox_to_anchor = (0, 0.9, 1, 0.2) if len(labels) < 3 else (0, 0.95, 1, 0.2)
    
    ax.legend(handles, labels,bbox_to_anchor=bbox_to_anchor, 
              loc="upper center",
              borderaxespad=0,
              frameon=False,
              ncol=3, 
              handlelength=1.5,
              shadow=False,
              fontsize = 13,
              handletextpad=0.1)
    
    if path is not None:
        filename = ylabel+'_iterations.pdf'
        fig.savefig(path+filename, bbox_inches='tight')
    
def set_tick_params(ax, 
                    wd=3, 
                    lt=5):

    ax.xaxis.set_tick_params(width=wd,length = lt)
    ax.yaxis.set_tick_params(width=wd,length = lt)
    
    ax.tick_params(axis='both', direction='in', which='major',labelsize=18)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.3)
        
def one_shot_baseline(ax, one_shot, max_iter=50):
    
    if one_shot is not None:
        for i in range(len(one_shot)):
            ax.plot([1,max_iter], [one_shot[i].mean(), one_shot[i].mean()], '--', label=('One-shot single-pass' if i==0 else '_'), alpha=0.2, color='grey')
            if i<=7:
                ax.text(x=1,y=one_shot[i].mean()-0.011,s='n='+str(100*(i+1)),fontsize=9)
        if i>7:
            ax.text(x=1,y=one_shot[i].mean()+0.003,s='n='+str(100*(i+1)),fontsize=9)

def plot_original_model(dataset,
                        model, 
                        X_train, 
                        y_train, 
                        path,
                        x_range=3.0):
    
    xx,yy = np.meshgrid(np.linspace(-x_range,x_range,200),np.linspace(-x_range,x_range,200))
    viz=np.c_[xx.ravel(),yy.ravel()]

    z = model.predict(viz)
    cc=[]
    for yt in y_train:
        if yt==1:
            cc.append('orchid')
        else:
            cc.append('mediumseagreen')
            
    plt.scatter(X_train[:, 0], X_train[:, 1], c=cc,  alpha=0.7)
    plt.imshow(z.reshape((200,200)), origin='lower', extent=(-x_range,x_range,-x_range,x_range),alpha=0.3, vmin=0, vmax=1, cmap = 'PiYG_r')
    plt.contour(xx,yy,z.reshape((200,200)),[0.5])

    plt.gcf().set_size_inches((6,6))
    plt.tick_params(
        axis='both',          # changes apply to both axes
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,         # ticks along the top edge are off
        right=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        labelleft=False) # labels along the bottom edge are off
    
    plt.savefig(os.path.join(path, 'data', dataset, 'original_model_colored.pdf'), bbox_inches='tight')
    plt.close()
    
def plot_copy_model(copy,
                    X_train,
                    y_train,
                    X_errors,
                    y_errors,
                    #t,
                    #run,
                    #path,
                    x_range=3.0):

    xx,yy = np.meshgrid(np.linspace(-x_range,x_range,200),np.linspace(-x_range,x_range,200))
    viz=np.c_[xx.ravel(),yy.ravel()]
    
    z = np.argmax(copy.predict(viz), axis=1)
    
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,  alpha=0.7)
    plt.scatter(X_errors[:, 0], X_errors[:, 1], c='red', marker='^', alpha=0.2)
    plt.imshow(z.reshape((200,200)), origin='lower', extent=(-x_range,x_range,-x_range,x_range),alpha=0.3, vmin=0, vmax=1)
    plt.contour(xx,yy,z.reshape((200,200)),[0.5])
    
    plt.xlim(-x_range,x_range)
    plt.ylim(-x_range,x_range)
    plt.autoscale(False)
    plt.gcf().set_size_inches((3,3))
    
    #plt.savefig(os.path.join(path, 'plots', f'RUN_{run}_ITER_{t}.pdf'), bbox_inches='tight')
    plt.show()