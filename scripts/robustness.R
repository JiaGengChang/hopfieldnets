library(dplyr)
library(tidyr)
library(ggbeeswarm)

x0 = read.table('loss_0.tsv',sep='\t',header = 1)['avg..acc']
x10 = read.table('loss_10.tsv',sep='\t',header = 1)['avg..acc']
x100 = read.table('loss_100.tsv',sep='\t',header = 1)['avg..acc']
x1000 = read.table('loss_1000.tsv',sep='\t',header = 1)['avg..acc']
x2000 = read.table('loss_2000.tsv',sep='\t',header = 1)['avg..acc']
x3000 = read.table('loss_3000.tsv',sep='\t',header = 1)['avg..acc']
x4000 = read.table('loss_4000.tsv',sep='\t',header = 1)['avg..acc']
x5000 = read.table('loss_5000.tsv',sep='\t',header = 1)['avg..acc']

df = tibble(n=1:50, 
            '0'=x0, 
            '10'=x10, 
            '100'=x100, 
            '1000'=x1000, 
            '2000'=x2000, 
            '3000'=x3000, 
            '4000'=x4000, 
            '5000'=x5000)

df[2:7] = df[2:7]*100

p = df %>% 
  pivot_longer(!n, names_to="num. loss") %>% 
  ggplot(aes(x=`num. loss`, y=value$avg..acc, color=`num. loss`)) + 
  geom_beeswarm(size=0.75) + 
  ggtitle("Random loss of weights vs accuracy") + 
  xlab("Number of weights lost") + 
  ylab("Average accuracy") + 
  theme(plot.title = element_text(size=20, hjust=0.5),
        axis.title.x = element_text(size=18),
        axis.title.y = element_text(size=18),
        axis.text.x = element_text(size=18),
        axis.text.y = element_text(size=18),
        legend.position="none"
  )


