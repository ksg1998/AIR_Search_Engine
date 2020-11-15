def permutationIndex():
	word_dict = read_from_disk("snippet.txt")
	file = open('all_permuterm.txt','w')
	terms = word_dict.keys()
	#print(terms)
	for term in sorted(terms):
		dkey = term + "$"
		for i in range(len(dkey),0,-1):
			out = rotate(dkey,i)
			file.write(out)
			file.write(" ")
			file.write(term)
			file.write('\n')
		
	file.close()


	
permutationIndex()
