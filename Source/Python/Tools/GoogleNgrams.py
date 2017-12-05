import phrasefinder
import sys

def GoogleNgrams(words, quiet=True):
    match = []
    volume = []

    word_count = len(words)
    counter = 1
    for x in words:

        if not quiet :
            sys.stdout.write("\r%d/%d" % (counter, word_count))
            sys.stdout.flush()

        match_str = '1'
        vol_str = '1'

        try:
            # search for term x through Google Ngrams using phrasefinder
            result = phrasefinder.search(x)

            if result.status == phrasefinder.Status.Ok:   
                if len(result.phrases) > 0:
                    match_str = (result.phrases[0].match_count)
                    vol_str = (result.phrases[0].volume_count)
        except:
            match_str = '-1'
            vol_str = '-1'

        match.append(match_str)
        volume.append(vol_str)

        counter += 1

    return match, volume
