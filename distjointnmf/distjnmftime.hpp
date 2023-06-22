/* Copyright 2022 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTJNMFTIME_HPP_
#define DISTNMF_DISTJNMFTIME_HPP_

/**
 * Class and function for collecting time statistics 
 */

namespace planc {

class DistJointNMFTime {
 private:
  double m_duration;
  double m_compute_duration;
  double m_communication_duration;
  double m_allgather_duration;
  double m_allreduce_duration;
  double m_reducescatter_duration;
  double m_sendrecv_duration;
  double m_gram_duration;
  double m_reg_duration;
  double m_nongram_duration;
  double m_mm_duration;
  double m_nnls_duration;
  double m_err_compute_duration;
  double m_err_communication_duration;
  
  // Matmul times
  double m_WtA_compute_duration;
  double m_WtA_communication_duration;
  double m_AH_compute_duration;
  double m_AH_communication_duration;
  double m_SHs_compute_duration;
  double m_SHs_communication_duration;

  // ANLS times
  double m_H2tS_compute_duration;
  double m_H2tS_communication_duration;

 public:
  DistJointNMFTime() {
    // Initialise all the breakdown timers
    this->m_duration = 0.0;
    this->m_compute_duration = 0.0;
    this->m_communication_duration = 0.0;
    this->m_allgather_duration = 0.0;
    this->m_allreduce_duration = 0.0;
    this->m_reducescatter_duration = 0.0;
    this->m_sendrecv_duration = 0.0;
    this->m_gram_duration = 0.0;
    this->m_reg_duration = 0.0;
    this->m_nongram_duration = 0.0;
    this->m_mm_duration = 0.0;
    this->m_nnls_duration = 0.0;
    this->m_err_compute_duration = 0.0;
    this->m_err_communication_duration = 0.0;

    // Matmul times
    this->m_WtA_compute_duration = 0.0;
    this->m_WtA_communication_duration = 0.0;
    this->m_AH_compute_duration = 0.0;
    this->m_AH_communication_duration = 0.0;
    this->m_SHs_compute_duration = 0.0;
    this->m_SHs_communication_duration = 0.0;

    // ANLS times
    this->m_H2tS_compute_duration = 0.0;
    this->m_H2tS_communication_duration = 0.0;
   }
  // Getter Functions
  const double duration() const { return m_duration; }
  const double compute_duration() const { return m_compute_duration; }
  const double communication_duration() const {
    return m_communication_duration;
  }
  const double allgather_duration() const { return m_allgather_duration; }
  const double allreduce_duration() const { return m_allreduce_duration; }
  const double reducescatter_duration() const {
    return m_reducescatter_duration;
  }
  const double sendrecv_duration() const { return m_sendrecv_duration; }
  const double gram_duration() const { return m_gram_duration; }
  const double reg_duration() const { return m_reg_duration; }
  const double nongram_duration() const { return m_nongram_duration; }
  const double mm_duration() const { return m_mm_duration; }
  const double nnls_duration() const { return m_nnls_duration; }
  const double err_compute_duration() const { return m_err_compute_duration; }
  const double err_communication_duration() const {
    return m_err_communication_duration;
  }
  const double WtA_compute_duration() const { return m_WtA_compute_duration; }
  const double WtA_communication_duration() const { 
    return m_WtA_communication_duration; 
  }
  const double AH_compute_duration() const { return m_AH_compute_duration; }
  const double AH_communication_duration() const { 
    return m_AH_communication_duration; 
  }
  const double SHs_compute_duration() const { return m_SHs_compute_duration; }
  const double SHs_communication_duration() const { 
    return m_SHs_communication_duration; 
  }
  const double H2tS_compute_duration() const {
    return m_H2tS_compute_duration; }
  const double H2tS_communication_duration() const { 
    return m_H2tS_communication_duration; 
  }
  // Update Functions
  void duration(double d) { m_duration += d; }
  void compute_duration(double d) { m_compute_duration += d; }
  void communication_duration(double d) { m_communication_duration += d; }
  void allgather_duration(double d) { m_allgather_duration += d; }
  void allreduce_duration(double d) { m_allreduce_duration += d; }
  void reducescatter_duration(double d) { m_reducescatter_duration += d; }
  void sendrecv_duration(double d) { m_sendrecv_duration += d; }
  void gram_duration(double d) { m_gram_duration += d; }
  void reg_duration(double d) { m_reg_duration += d; }
  void nongram_duration(double d) { m_nongram_duration += d; }
  void mm_duration(double d) { m_mm_duration += d; }
  void nnls_duration(double d) { m_nnls_duration += d; }
  void err_compute_duration(double d) { m_err_compute_duration += d; }
  void err_communication_duration(double d) {
    m_err_communication_duration += d;
  }
  void WtA_compute_duration(double d) { m_WtA_compute_duration += d; }
  void WtA_communication_duration(double d) { 
    m_WtA_communication_duration += d; 
  }
  void AH_compute_duration(double d) { m_AH_compute_duration += d; }
  void AH_communication_duration(double d) { 
    m_AH_communication_duration += d; 
  }
  void SHs_compute_duration(double d) { m_SHs_compute_duration += d; }
  void SHs_communication_duration(double d) { 
    m_SHs_communication_duration += d; 
  }
  void H2tS_compute_duration(double d) { m_H2tS_compute_duration += d; }
  void H2tS_communication_duration(double d) { 
    m_H2tS_communication_duration += d; 
  }
};

}  // namespace planc

#endif  // DISTNMF_DISTNMFTIME_HPP_
