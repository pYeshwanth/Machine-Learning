def count_vowels_consonants(s):
    vowels = "aeiouAEIOU"
    v=0
    c=0

    for ch in s:
        if ch.isalpha():
            if ch in vowels :
                 v = v+1
            else :
                c = c+1
    return v,c

text = input("Enter a String : ")
vowels, consonants = count_vowels_consonants(text)
print("Vowels: ",vowels)
print("Consonants: ",consonants)
