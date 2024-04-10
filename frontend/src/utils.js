export function strNumber(num) {
    if (num > 1e14) {
      return `${(num / 1e12).toFixed(0)}T`;
    } else if (num > 1e12) {
      return `${(num / 1e12).toFixed(1)}T`;
    } else if (num > 1e11) {
      return `${(num / 1e9).toFixed(0)}G`;
    } else if (num > 1e9) {
      return `${(num / 1e9).toFixed(1)}G`;
    } else if (num > 1e8) {
      return `${(num / 1e6).toFixed(0)}M`;
    } else if (num > 1e6) {
      return `${(num / 1e6).toFixed(1)}M`;
    } else if (num > 1e5) {
      return `${(num / 1e3).toFixed(0)}K`;
    } else if (num > 1e3) {
      return `${(num / 1e3).toFixed(1)}K`;
    } else if (num >= 1) {
      return `${num.toFixed(1)}`;
    } else {
      return `${num.toFixed(2)}`;
    }
  }

  export function strNumber_1024(num) {
    if (num > (2**40)) {
      return `${(num /2**40 ).toFixed(1)}T`;
    } else if (num > (2**30)) {
      return `${(num /2**30 ).toFixed(1)}G`;
    } else if (num > (2**20)) {
      return `${(num /2**20 ).toFixed(1)}M`;
    } else if (num > (2**10)) {
      return `${(num /2**10 ).toFixed(1)}K`;
    } else {
      return `${num.toFixed(1)}`;
    }
  }

  
  export function strNumberTime(num) {
    if (num >= 1) {
      return `${num.toFixed(1)}s`;
    } else if (num > 1e-3) {
      return `${(num * 1e3).toFixed(1)}ms`;
    } else if (num > 1e-6) {
      return `${(num * 1e6).toFixed(1)}us`;
    } else if (num > 1e-9) {
      return `${(num * 1e9).toFixed(1)}ns`;
    } else {
      return `${num.toFixed(0)}s`;
    }
  }