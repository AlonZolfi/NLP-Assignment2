import ex2
import time
# text = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat.'
sp = ex2.Spell_Checker()
sp.build_model(open('test_corpus.txt','r').read())

s = time.time()
sp.learn_error_tables('common_errors.txt')
e = time.time()
print(e-s)

print(sp.spell_check('i love my famly', 0.9))
print(sp.spell_check('i love my family', 0.9))