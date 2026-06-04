class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.buffer = [];
    this.sourcePosition = 0;
    this.ratio = sampleRate / this.targetRate;
    this.port.onmessage = (event) => {
      if (event.data === "flush") this.flush();
    };
  }

  send(samples) {
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      pcm[i] = value < 0 ? value * 32768 : value * 32767;
    }
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
  }

  flush() {
    if (this.buffer.length) this.send(this.buffer.splice(0));
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels.length || !channels[0].length) return true;
    const inputLength = channels[0].length;

    while (this.sourcePosition < inputLength - 1) {
      const index = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - index;
      let sample = 0;
      for (const channel of channels) {
        sample += channel[index] + (channel[index + 1] - channel[index]) * fraction;
      }
      this.buffer.push(sample / channels.length);
      this.sourcePosition += this.ratio;
    }
    this.sourcePosition -= inputLength;

    while (this.buffer.length >= 2048) this.send(this.buffer.splice(0, 2048));
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
