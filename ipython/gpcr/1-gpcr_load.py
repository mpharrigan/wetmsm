
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import tables
import glob
from wetmsm.analysis import SolventShellsAnalysis


# In[3]:

cd ~/biox/implement/water/gpcr/


# In[4]:

seqs = []
fns = []
for count_fn in glob.iglob('2rh1/generation2/trr/*/clone*.trr.pop.count.h5'):
    count_f = tables.open_file(count_fn)
    count_a = np.array(count_f.root.shell_counts)
    
    if len(count_a) > 1:
        seqs.append(count_a)
        fns.append(count_fn)
    print(count_a.shape)
    count_f.close()
    


# In[5]:

sa_pop = SolventShellsAnalysis(seqs, 0.2)


# In[6]:

plt.plot(sa_pop.seqs2d[0][:,13:15])


# In[7]:

sa_pop.deleted


# In[8]:

cd ~/implement/wetmsm/gpcr/


# In[9]:

import pickle


# In[10]:

with open('1-pop.pickl', 'wb') as f:
    pickle.dump(seqs, f)


# In[11]:

with open('1-pop-fns.pickl', 'wb') as f:
    pickle.dump(fns, f)


# # tICA

# In[13]:

sa_pop.fit_tica(lag_time=1)


# In[17]:

with open('2-pop-tica.pickl', 'wb') as f:
    pickle.dump(sa_pop.tica, f)


# In[18]:

with open('3-pop-ticax.pickl', 'wb') as f:
    pickle.dump(sa_pop.ticax, f)


# In[14]:

plt.subplots(figsize=(3,5))
plt.hlines(sa_pop.tica.timescales_, 0, 1, 'b')


# In[15]:

txy = np.concatenate(sa_pop.ticax)


# In[22]:

plt.scatter(txy[:,0], txy[:,1], linewidth=0, s=1, alpha=0.1)
plt.plot(sa_pop.ticax[1848][:,0], sa_pop.ticax[1848][:,1], 'ro-')


# In[21]:

num = 0
for i, tx in enumerate(sa_pop.ticax):
    if np.any(tx[:,0]<-25) and np.any(tx[:,0] > 40):
        print(i, np.min(tx[:,0]), np.max(tx[:,0]), tx.shape)
        num += 1
print(num)


# In[24]:

fns[834]


# # VMD

# In[65]:

from wetmsm.vmd_write import VMDWriter, VMDSCRIPT
from os.path import join as pjoin


# ### Load Assignments File

# In[47]:

biox_prefix = '/home/harrigan/biox/implement/water/gpcr/2rh1/generation2'
assn_fn = pjoin(biox_prefix, 'trr/4/clone4397.trr.pop.assign.h5')
assn_f = tables.open_file(assn_fn)
assn = assn_f.root.assignments
print(assn.shape)


# ### Load Indices

# In[49]:

ca_inds = np.loadtxt(pjoin(biox_prefix, 'ca_2rh1.dat'), dtype=int)
pop_inds = np.loadtxt(pjoin(biox_prefix, 'pop_2rh1.dat'), dtype=int)
print(ca_inds.shape, pop_inds.shape)


# ### Load topology

# In[53]:

import mdtraj as md
top = md.load('vis/2rh1-ticaleftright/FullSystem.ions.eq2.out.2rh1.pdb')
print(top)


# ### Visualize loadings

# In[31]:

tic1 = sa_pop.tica.components_[0]
xx = np.arange(len(tic1))


# In[45]:

plt.subplots(2,2, figsize=(10,6))
plt.subplot(2,2,1)
plt.scatter(xx,tic1, linewidth=0, s=30)
plt.subplot(2,2,2)
tic1s = tic1 ** 2
tic1s /= np.max(tic1s)
plt.scatter(xx, tic1s, linewidth=0, s=30, c='r')
loading1 = np.copy(tic1s)
loading1[tic1s < 0.1] = 0.0
plt.subplot(2,2,3)
plt.scatter(xx, loading1, linewidth=0, s=30, c='purple')


# ### Make object and compute

# In[55]:

i = 834
vmd = VMDWriter(assn=assn, solvent_ind=pop_inds, n_frames=sa_pop.ticax[i].shape[0], n_atoms=top.n_atoms,
                n_solute=len(ca_inds), n_shells=sa_pop.seqs3d[i].shape[2])


# In[59]:

user = vmd.compute(loading1, sa_pop.deleted)


# ### Save

# In[64]:

np.savetxt('vis/2rh1-ticaleftright/pop.dat', user, fmt='%.5f')


# ### Generate script

# In[67]:

with open('vis/2rh1-ticaleftright/pop.tcl', 'w') as f:
    f.write(VMDSCRIPT.format(top_fn='FullSystem.ions.eq2.out.2rh1.pdb', traj_fn='clone4397.trr', step=1, dat_fn='pop.dat'))


# In[ ]:



