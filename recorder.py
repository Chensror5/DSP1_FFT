import sounddevice as sd
import soundfile as sf


FS = 16000
sd.default.device=2
sd.default.samplerate=FS

print('Recording...')
rec = sd.rec(FS*10,FS,1)
sd.wait()
sf.write('out.wav',rec,FS)

