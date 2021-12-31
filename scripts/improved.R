library(dplyr)
library(tidyr)
library(forcats)
library(ggbeeswarm)

x5 = read.table('g50_n5.tsv',sep='\t',header=1)['avg..acc']
x10 = read.table('g50_n10.tsv',sep='\t',header=1)['avg..acc']
x15 = read.table('g50_n15.tsv',sep='\t',header=1)['avg..acc']
x20 = read.table('g50_n20.tsv',sep='\t',header=1)['avg..acc']
x25 = read.table('g50_n25.tsv',sep='\t',header=1)['avg..acc']
x30 = read.table('g50_n30.tsv',sep='\t',header=1)['avg..acc']

df = tibble(n=1:50,
            'T=05'=x5,
            'T=10'=x10,
            'T=15'=x15,
            'T=20'=x20,
            'T=25'=x25,
            'T=30'=x30)

#scale accuracy to 100

df[2:7] = df[2:7]*100

p = df %>% 
  pivot_longer(!n, names_to="num. loss") %>% 
  ggplot(aes(x=`num. loss`, y=value, color=`num. loss`)) + 
  geom_quasirandom(size=1.5) +
  ggtitle("(b) With 50 steps of gradient descent") + 
  xlab("Number of training vectors") + 
  ylab("Average accuracy") + 
  theme(plot.title = element_text(size=20, hjust=0.5),
        axis.title.x = element_text(size=18),
        axis.title.y = element_text(size=18),
        axis.text.x = element_text(size=18, angle=45, hjust=1),
        axis.text.y = element_text(size=18),
        legend.position="none"
  ) + stat_summary(fun = median, 
                   fun.min = function(x){quantile(x, 0.25)},
                   fun.max = function(x){quantile(x, 0.75)},
                   geom="crossbar", width=0.5, color='black')

ggsave("gd50.png", p, height=6, width=12, units="in")
