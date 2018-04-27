plt.figure(figsize=(6,4))

plt.suptitle(r'$f_{\rm sky}=0.33 \quad M > 6\times 10^{13}M_{\odot} \quad 0.6 < z < 1.0$')#' + FG')

ax1 = plt.subplot(121)

ax1.set_title(r'$\Delta_T = 3.0\mu$K-arcmin')

# fishy_kSZ[3.0][3.0].plot('w','mnu', tag=r'$D=3$m', color='#555B6E', howmanysigma=[1])
# fishy_kSZ[3.0][4.0].plot('w','mnu', tag=r'$D=4$m', color='#89B0AE', howmanysigma=[1])
# fishy_kSZ[3.0][5.0].plot('w','mnu', tag=r'$D=5$m', color='#BEE3DB', howmanysigma=[1])

fishy_kSZ[3.0][3.0].plot('w','mnu', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
fishy_kSZ[3.0][4.0].plot('w','mnu', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
fishy_kSZ[3.0][5.0].plot('w','mnu', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
fishy_kSZ[3.0][6.0].plot('w','mnu', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('w','mnu', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

fishy_kSZ_fg[3.0][3.0].plot('w','mnu', color='#89CE94', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][4.0].plot('w','mnu', color='#86A59C', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][5.0].plot('w','mnu', color='#7D5BA6', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][6.0].plot('w','mnu', color='#643173', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][7.0].plot('w','mnu', color='#333333', howmanysigma=[1], fill=0, ls='--')

ax1.set_ylim(0,0.19)
ax1.set_xlim([-1.5,-0.5])
ax1.legend(frameon=False)
ax1.set_ylabel(r'$M_{\nu}$ [eV]', size=10)


# plt.subplot(122)
ax2 = plt.subplot(122, sharex=ax1)

ax2.set_title(r'$D=7$ m')

plt.plot(1e3,1e3,label=r'CMB', color='#8980F5')
fishy_kSZ[5.0][7.0].fcmb.plot('w','mnu', tag=r'CMB', color='#8980F5', howmanysigma=[1])
fishy_kSZ[5.0][7.0].plot('w','mnu', tag=r'$\Delta_T=5\mu$K-arcmin', color='#BEE3DB', howmanysigma=[1])
fishy_kSZ[4.0][7.0].plot('w','mnu', tag=r'$\Delta_T=4\mu$K-arcmin', color='#89B0AE', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('w','mnu', tag=r'$\Delta_T=3\mu$K-arcmin', color='#555B6E', howmanysigma=[1])

fishy_kSZ_fg[5.0][7.0].plot('w','mnu', color='#BEE3DB', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[4.0][7.0].plot('w','mnu', color='#89B0AE', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][7.0].plot('w','mnu', color='#555B6E', howmanysigma=[1], fill=0, ls='--')

ax2.set_ylim(0,0.19)
ax2.set_xlim([-1.5,-0.5])
ax2.set_ylabel('')
plt.setp(ax2.get_yticklabels(), visible=False)

ax2.legend(frameon=False)

plt.subplots_adjust(wspace=0, hspace=0, top=0.86)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~``
plt.figure(figsize=(6,4))

plt.suptitle(r'$f_{\rm sky}=0.33 \quad M > 6\times 10^{13}M_{\odot} \quad 0.6 < z < 1.0$')#' + FG + $b_{\tau}$ weak prior')

ax1 = plt.subplot(121)

ax1.set_title(r'$\Delta_T = 3.0\mu$K-arcmin')

# fishy_kSZ[3.0][3.0].plot('w','gamma0', tag=r'$D=3$m', color='#C5E6A6', howmanysigma=[1])
# fishy_kSZ[3.0][4.0].plot('w','gamma0', tag=r'$D=4$m', color='#BDD2A6', howmanysigma=[1])
# fishy_kSZ[3.0][5.0].plot('w','gamma0', tag=r'$D=5$m', color='#B9BEA5', howmanysigma=[1])
# fishy_kSZ[3.0][6.0].plot('w','gamma0', tag=r'$D=6$m', color='#A7AAA4', howmanysigma=[1])
# fishy_kSZ[3.0][7.0].plot('w','gamma0', tag=r'$D=7$m', color='#9899A6', howmanysigma=[1])

fishy_kSZ[3.0][3.0].plot('w','gamma0', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
fishy_kSZ[3.0][4.0].plot('w','gamma0', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
fishy_kSZ[3.0][5.0].plot('w','gamma0', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
fishy_kSZ[3.0][6.0].plot('w','gamma0', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('w','gamma0', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

fishy_kSZ_fg[3.0][3.0].plot('w','gamma0', color='#89CE94', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][4.0].plot('w','gamma0', color='#86A59C', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][5.0].plot('w','gamma0', color='#7D5BA6', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][6.0].plot('w','gamma0', color='#643173', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][7.0].plot('w','gamma0', color='#333333', howmanysigma=[1], fill=0, ls='--')

ax1.legend(frameon=False)
ax1.set_ylim([0.45,0.65])
ax1.set_xlim([-1.5,-0.5])

# plt.subplot(122)
ax2 = plt.subplot(122, sharex=ax1)

ax2.set_title(r'$D=7$ m')

fishy_kSZ[5.0][7.0].plot('w','gamma0', tag=r'$\Delta_T=5\mu$K-arcmin', color='#BEE3DB', howmanysigma=[1])
fishy_kSZ[4.0][7.0].plot('w','gamma0', tag=r'$\Delta_T=4\mu$K-arcmin', color='#89B0AE', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('w','gamma0', tag=r'$\Delta_T=3\mu$K-arcmin', color='#555B6E', howmanysigma=[1])

fishy_kSZ_fg[5.0][7.0].plot('w','gamma0', color='#BEE3DB', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[4.0][7.0].plot('w','gamma0', color='#89B0AE', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][7.0].plot('w','gamma0', color='#555B6E', howmanysigma=[1], fill=0, ls='--')

ax2.set_ylim([0.45,0.65])
ax2.set_xlim([-1.5,-0.5])
ax2.set_ylabel('')
plt.setp(ax2.get_yticklabels(), visible=False)

ax2.legend(frameon=False)

plt.subplots_adjust(wspace=0, hspace=0, top=0.86)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~``
plt.figure(figsize=(6,4))

plt.suptitle(r'$f_{\rm sky}=0.33 \quad M > 6\times 10^{13}M_{\odot} \quad 0.6 < z < 1.0$')#' + FG + $b_{\tau}$ weak prior')

ax1 = plt.subplot(121)

ax1.set_title(r'$\Delta_T = 3.0\mu$K-arcmin')

# fishy_kSZ[3.0][3.0].plot('mnu','gamma0', tag=r'$D=3$m', color='#555B6E', howmanysigma=[1])
# fishy_kSZ[3.0][4.0].plot('mnu','gamma0', tag=r'$D=4$m', color='#89B0AE', howmanysigma=[1])
# fishy_kSZ[3.0][5.0].plot('mnu','gamma0', tag=r'$D=5$m', color='#BEE3DB', howmanysigma=[1])

fishy_kSZ[3.0][3.0].plot('mnu','gamma0', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
fishy_kSZ[3.0][4.0].plot('mnu','gamma0', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
fishy_kSZ[3.0][5.0].plot('mnu','gamma0', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
fishy_kSZ[3.0][6.0].plot('mnu','gamma0', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('mnu','gamma0', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

fishy_kSZ_fg[3.0][3.0].plot('mnu','gamma0', color='#89CE94', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][4.0].plot('mnu','gamma0', color='#86A59C', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][5.0].plot('mnu','gamma0', color='#7D5BA6', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][6.0].plot('mnu','gamma0', color='#643173', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][7.0].plot('mnu','gamma0', color='#333333', howmanysigma=[1], fill=0, ls='--')


ax1.legend(frameon=False)

ax1.set_xlim([0.,0.2])
ax1.set_ylim([0.45,0.65])
ax1.set_xlabel(r'$M_{\nu}$ [eV]', size=10)

ax2 = plt.subplot(122, sharex=ax1)

ax2.set_title(r'$D=7$ m')

fishy_kSZ[5.0][5.0].plot('mnu','gamma0', tag=r'$\Delta_T=5\mu$K-arcmin', color='#BEE3DB', howmanysigma=[1])
fishy_kSZ[4.0][5.0].plot('mnu','gamma0', tag=r'$\Delta_T=4\mu$K-arcmin', color='#89B0AE', howmanysigma=[1])
fishy_kSZ[3.0][5.0].plot('mnu','gamma0', tag=r'$\Delta_T=3\mu$K-arcmin', color='#555B6E', howmanysigma=[1])

fishy_kSZ_fg[5.0][7.0].plot('mnu','gamma0', color='#BEE3DB', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[4.0][7.0].plot('mnu','gamma0', color='#89B0AE', howmanysigma=[1], fill=0, ls='--')
fishy_kSZ_fg[3.0][7.0].plot('w','gamma0', color='#555B6E', howmanysigma=[1], fill=0, ls='--')

ax2.set_ylabel('')
plt.setp(ax2.get_yticklabels(), visible=False)

ax2.set_xlim([0.,0.2])
ax2.set_ylim([0.45,0.65])
ax2.set_xlabel(r'$M_{\nu}$ [eV]', size=10)

ax2.legend(frameon=False)

plt.subplots_adjust(wspace=0, hspace=0, top=0.86)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure(figsize=(5,5))

plt.suptitle(r'$f_{\rm sky}=0.33 \quad M > 6\times 10^{13}M_{\odot} \quad 0 < z < 0.6 \quad \Delta_T = 3.0\mu$K-$^{\prime}$')#' + FG')

ax1 = plt.subplot(221)

# ax1.set_title(r'$\Delta_T = 3.0\mu$K-arcmin')

fishy_kSZ[5.0][7.0].fcmb.plot('w','mnu', color='#8980F5', howmanysigma=[1])
fishy_kSZ[3.0][3.0].plot('w','mnu', color='#89CE94', howmanysigma=[1])
fishy_kSZ[3.0][4.0].plot('w','mnu', color='#86A59C', howmanysigma=[1])
fishy_kSZ[3.0][5.0].plot('w','mnu', color='#7D5BA6', howmanysigma=[1])
fishy_kSZ[3.0][6.0].plot('w','mnu', color='#643173', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('w','mnu', color='#333333', howmanysigma=[1])

# fishy_kSZ_fg[3.0][3.0].plot('w','mnu', color='#89CE94', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][4.0].plot('w','mnu', color='#86A59C', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][5.0].plot('w','mnu', color='#7D5BA6', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][6.0].plot('w','mnu', color='#643173', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][7.0].plot('w','mnu', color='#333333', howmanysigma=[1], fill=0, ls='--')

ax1.set_xlim([-1.5,-0.5])
ax1.set_ylim([0.,0.2])
ax1.set_ylabel(r'$M_{\nu}$ [eV]', size=10)
ax1.set_xticks([-1.])
plt.setp(ax1.get_xticklabels(), visible=False)

axl = plt.subplot(222)

plt.plot([1e3,1e3], label=r'CMB only', color='#8980F5')#, howmanysigma=[1])
plt.plot([1e3,1e3], label=r'$D=3$m', color='#89CE94')#, howmanysigma=[1])
plt.plot([1e3,1e3], label=r'$D=4$m', color='#86A59C')#, howmanysigma=[1])
plt.plot([1e3,1e3], label=r'$D=5$m', color='#7D5BA6')#, howmanysigma=[1])
plt.plot([1e3,1e3], label=r'$D=6$m', color='#643173')#, howmanysigma=[1])
plt.plot([1e3,1e3], label=r'$D=7$m', color='#333333')#, howmanysigma=[1])
# plt.plot([1e3,1e3], label=r'w/ FG @ 150 GHz', color='k', ls='--')#, howmanysigma=[1])

axl.set_xlim([0.,0.2])
axl.set_ylim([0.45,0.65])
axl.set_ylabel('')
axl.set_xlabel('')
axl.legend(frameon=False)
plt.setp(axl.get_yticklabels(), visible=False)
plt.setp(axl.get_xticklabels(), visible=False)


ax2 = plt.subplot(223)#, sharex=ax1)

fishy_kSZ[3.0][3.0].plot('w','gamma0', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
fishy_kSZ[3.0][4.0].plot('w','gamma0', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
fishy_kSZ[3.0][5.0].plot('w','gamma0', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
fishy_kSZ[3.0][6.0].plot('w','gamma0', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('w','gamma0', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

# fishy_kSZ_fg[3.0][3.0].plot('w','gamma0', color='#89CE94', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][4.0].plot('w','gamma0', color='#86A59C', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][5.0].plot('w','gamma0', color='#7D5BA6', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][6.0].plot('w','gamma0', color='#643173', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][7.0].plot('w','gamma0', color='#333333', howmanysigma=[1], fill=0, ls='--')

ax2.set_ylim([0.45,0.65])
ax2.set_xlim([-1.5,-0.5])
ax2.set_yticks([0.45,0.5,0.55,0.6])
ax2.set_xticks([-1.5,-1.25,-1.,-0.75])

ax3 = plt.subplot(224, sharey=ax2)

fishy_kSZ[3.0][3.0].plot('mnu','gamma0', tag=r'$D=3$m', color='#89CE94', howmanysigma=[1])
fishy_kSZ[3.0][4.0].plot('mnu','gamma0', tag=r'$D=4$m', color='#86A59C', howmanysigma=[1])
fishy_kSZ[3.0][5.0].plot('mnu','gamma0', tag=r'$D=5$m', color='#7D5BA6', howmanysigma=[1])
fishy_kSZ[3.0][6.0].plot('mnu','gamma0', tag=r'$D=6$m', color='#643173', howmanysigma=[1])
fishy_kSZ[3.0][7.0].plot('mnu','gamma0', tag=r'$D=7$m', color='#333333', howmanysigma=[1])

# fishy_kSZ_fg[3.0][3.0].plot('mnu','gamma0', color='#89CE94', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][4.0].plot('mnu','gamma0', color='#86A59C', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][5.0].plot('mnu','gamma0', color='#7D5BA6', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][6.0].plot('mnu','gamma0', color='#643173', howmanysigma=[1], fill=0, ls='--')
# fishy_kSZ_fg[3.0][7.0].plot('mnu','gamma0', color='#333333', howmanysigma=[1], fill=0, ls='--')

ax3.set_xlim([0.,0.2])
ax3.set_ylim([0.45,0.65])
ax3.set_xlabel(r'$M_{\nu}$ [eV]', size=10)
ax3.set_ylabel('')
plt.setp(ax3.get_yticklabels(), visible=False)


plt.subplots_adjust(wspace=0, hspace=0, top=0.86)

